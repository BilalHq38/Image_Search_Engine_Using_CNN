import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pickle
import hashlib
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, send_from_directory, jsonify
from torchvision import models, transforms
import torch
import torch.nn as nn
import clip
from tqdm import tqdm
import faiss
from torchvision.models import ResNet50_Weights

# --- Config & Flask ---
app = Flask(__name__, template_folder="templates", static_folder="static")

IMAGE_FOLDER = os.path.join(app.static_folder, "images")
UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
CACHE_PATH = 'image_features.pkl'

os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Models ---
# ResNet50 backbone -> feature map (remove last 2 layers)
resnet_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet_model = nn.Sequential(*list(resnet_model.children())[:-2])
resnet_model.eval().to(device)

# CLIP
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# ResNet preprocessing
transform_resnet = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# --- Utility Functions ---
def compute_image_hash(image_path: str) -> str:
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def extract_clip_features(image: Image.Image) -> np.ndarray:
    # simple test-time augmentation: flip
    views = [image, image.transpose(Image.FLIP_LEFT_RIGHT)]
    feats = []
    for view in views:
        image_tensor = clip_preprocess(view).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = clip_model.encode_image(image_tensor).squeeze()
        feats.append(emb.detach().cpu().numpy())
    return np.mean(feats, axis=0)

def extract_resnet_features(image: Image.Image) -> np.ndarray:
    image_tensor = transform_resnet(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat_map = resnet_model(image_tensor)  # [B, C=2048, H, W]
        avg_pool = torch.nn.functional.adaptive_avg_pool2d(feat_map, (1, 1)).squeeze()  # 2048
        max_pool = torch.nn.functional.adaptive_max_pool2d(feat_map, (1, 1)).squeeze()  # 2048
        combined = torch.cat((avg_pool, max_pool), dim=0)  # 4096
    return combined.detach().cpu().numpy()

def extract_color_histogram(image: Image.Image, bins=(8, 8, 8)) -> np.ndarray:
    image_hsv = image.convert('HSV')
    image_np = np.array(image_hsv)  # H, W, 3
    hist, _ = np.histogramdd(
        image_np.reshape(-1, 3),
        bins=bins,
        range=[(0, 256), (0, 256), (0, 256)]
    )
    hist_flat = hist.flatten().astype(np.float32)
    norm = np.linalg.norm(hist_flat) or 1.0
    return hist_flat / norm  # 8*8*8=512

def extract_combined_features(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert('RGB')
    clip_feat = extract_clip_features(image)
    cnn_feat = extract_resnet_features(image)
    color_feat = extract_color_histogram(image)

    # Normalize each block before concat
    def nz_norm(x):
        n = np.linalg.norm(x)
        return x / (n if n != 0 else 1.0)

    clip_norm = nz_norm(clip_feat)        # 512
    cnn_norm = nz_norm(cnn_feat)          # 4096
    color_norm = nz_norm(color_feat)      # 512

    combined = np.concatenate((clip_norm, cnn_norm, color_norm)).astype(np.float32)  # 5120
    # Normalize final vector for cosine via inner product
    final_norm = np.linalg.norm(combined) or 1.0
    return combined / final_norm

# --- Cache & FAISS ---
image_embeddings = {}     # {img_hash: {"features": np.ndarray, "filename": str}}
filename_list = []        # [(img_hash, filename), ...]
faiss_index = None
dim = 512 + 4096 + 512    # 5120

def load_or_build_index():
    global image_embeddings, filename_list, faiss_index

    # Create an IP index (cosine similarity if vectors are normalized)
    index = faiss.IndexFlatIP(dim)

    # Load existing cache if present
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'rb') as f:
            image_embeddings = pickle.load(f)

    # Scan image folder, add new files to cache
    all_files = [f for f in os.listdir(IMAGE_FOLDER)
                 if os.path.isfile(os.path.join(IMAGE_FOLDER, f))]

    new_files = []
    for fname in all_files:
        path = os.path.join(IMAGE_FOLDER, fname)
        img_hash = compute_image_hash(path)
        if img_hash not in image_embeddings:
            new_files.append((img_hash, fname))

    if new_files:
        print(f"🔄 Updating cache with {len(new_files)} new images...")
        for img_hash, fname in tqdm(new_files):
            try:
                path = os.path.join(IMAGE_FOLDER, fname)
                feat = extract_combined_features(path)
                image_embeddings[img_hash] = {'features': feat, 'filename': fname}
            except Exception as e:
                print(f"Skipping {fname}: {e}")

        # Save updated cache
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump(image_embeddings, f)

    # Prepare FAISS index
    filename_list = []
    vectors = []
    for img_hash, data in image_embeddings.items():
        vec = data['features']  # already normalized
        vectors.append(vec.astype(np.float32))
        filename_list.append((img_hash, data['filename']))

    if vectors:
        index.add(np.vstack(vectors))

    faiss_index = index

# Build index on startup
load_or_build_index()

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    if 'query_image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['query_image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # save uploaded file
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    # number of desired results (default 20)
    try:
        top_k = int(request.form.get("result_count", 20))
    except Exception:
        top_k = 20
    top_k = max(1, min(1000, top_k))  # clamp

    results = []
    note = ""
    query_hash = compute_image_hash(save_path)

    # If exact image already indexed, include it
    if query_hash in image_embeddings:
        exact = image_embeddings[query_hash]
        results.append({
            "filename": exact['filename'],
            "path": f"/static/images/{exact['filename']}",
            "score": "1.0000",
            "modified": os.path.getmtime(os.path.join(IMAGE_FOLDER, exact['filename'])),
            "exact": True
        })
        note = "Exact match found by hash"

    # Similarity search (only if we have an index with vectors)
    if faiss_index is not None and faiss_index.ntotal > 0:
        query_vec = extract_combined_features(save_path).astype(np.float32).reshape(1, -1)
        # Make sure normalized (safety)
        norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
        query_vec = query_vec / np.where(norm == 0, 1.0, norm)

        # Request a bit more to allow filtering duplicates
        want = min(top_k + 5, faiss_index.ntotal)
        if want > 0:
            D, I = faiss_index.search(query_vec, want)

            seen_hashes = {query_hash} if query_hash in image_embeddings else set()
            for idx, score in zip(I[0], D[0]):
                if idx < 0 or idx >= len(filename_list):
                    continue
                img_hash, fname = filename_list[idx]
                # avoid duplicate if exact match coincides
                if img_hash in seen_hashes:
                    continue
                seen_hashes.add(img_hash)
                results.append({
                    "filename": fname,
                    "path": f"/static/images/{fname}",
                    "score": f"{float(score):.4f}",
                    "modified": os.path.getmtime(os.path.join(IMAGE_FOLDER, fname)),
                    "exact": False
                })
                if len(results) >= top_k + (1 if note else 0):
                    break
    else:
        note = note or "Index empty — add images to /static/images"

    # If there was no exact match and still nothing similar, results may be empty
    return jsonify({"results": results, "note": note})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Optional: simple endpoint to rebuild the index (if you add/remove images)
@app.route('/reindex', methods=['POST'])
def reindex():
    load_or_build_index()
    return jsonify({"status": "ok", "total_indexed": faiss_index.ntotal if faiss_index else 0})

if __name__ == '__main__':
    # Run: python app.py
    app.run(host='0.0.0.0', port=5050, debug=True)

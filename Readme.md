🖼️ Image Search Engine
A powerful, multi-featured Image Search Engine built with Flask, PyTorch, CLIP, ResNet, and FAISS, capable of finding visually similar images based on deep features and color histograms.

📸 Features
🔍 Search by Image: Upload an image and find visually similar results.
🤖 Multi-modal Feature Extraction:
CLIP ViT-B/32 (semantic embeddings)
ResNet50 (deep vision features)
Color Histograms (HSV)

⚡ Fast Similarity Search using FAISS (cosine similarity)
💾 Caching: Embeddings are saved to speed up startup.
🔒 Password Protected UI
🎨 Dark Mode, Tailwind CSS UI, and responsive layout
🔁 Live Preview & Modal View for search results
🚀 Ready to deploy on localhost or cloud services

🧠 How It Works

🏗 Feature Extraction
Each image is transformed into a 5120-dimensional vector:
512 from CLIP (ViT-B/32)
4096 from ResNet50 (average + max pooled features)
512 from HSV color histogram (8x8x8)

🔍 Similarity Search
Vectors are L2-normalized and stored in a FAISS IndexFlatIP (Inner Product Index).
When an image is uploaded, its embedding is extracted and compared to others in the index.
Optionally detects exact matches using MD5 hash.

🗂 Caching
Feature vectors and metadata are cached in image_features.pkl to avoid re-processing.

🧰 Requirements
Install Python dependencies:
pip install -r requirements.txt
requirements.txt

🔧 If using GPU, replace faiss-cpu with faiss-gpu.

🖥️ Running the App

Place your reference images in the folder:
static/images/

Run the Flask app:
python app.py

Open your browser and go to:
http://localhost:5050

🔐 UI Password Protection
When the page loads, users are prompted to enter a password.
Default password: password

To change it, update this line in the HTML:
const CORRECT_PASSWORD = 'password'; // Change this

🧪 Usage Guide

🔎 Searching for Similar Images
Go to the home page.
Upload a query image (JPG, PNG, etc.)
Set the number of desired results (1–250).
Click Search Images.
Results will appear with thumbnails, scores, and match type.

🔁 Rebuilding the Index
If you add or remove images in static/images, click the Reindex endpoint via:
curl -X POST http://localhost:5050/reindex
Or trigger it via a browser with tools like Postman.

🗃 Folder Structure
├── app.py                    # Flask backend
├── image_features.pkl        # Feature cache (auto-generated)
├── requirements.txt
├── static/
│   ├── images/               # Reference images
│   └── uploads/              # Uploaded query images
├── templates/
│   └── index.html            # Frontend HTML

📸 Example Use Cases
Product image search (e.g. “find shoes like this”)
Art or visual style similarity
Duplicate image detection
Color-based photo organization

🌐 Deployment Notes
Designed to run on localhost, but can be easily deployed using:
Docker
Heroku
AWS EC2
Railway.app

Make sure to adjust the UPLOAD_FOLDER and IMAGE_FOLDER paths if deploying to a different environment.
Set a strong password in production!

✅ To-Do / Enhancements
 Add support for reverse color search only
 Enable GPU toggle dynamically
 Add pagination or infinite scroll
 Use CLIP text queries (zero-shot)
 User image gallery

📃 License
MIT License — use it freely, commercially, and open-source.

🙌 Acknowledgments
OpenAI CLIP
FAISS by Facebook AI
Torchvision Models
Tailwind CSS
## Image Search Engine Using CNN

A compact, self-hosted image search engine built with Flask and deep visual embeddings. It uses CLIP and ResNet for feature extraction plus HSV color histograms and FAISS for fast similarity search.

### Key features
- Search by image (upload an image to find visually similar images)
- Multi-modal embeddings: CLIP (ViT-B/32), ResNet50, and color histograms (HSV)
- Fast nearest-neighbor search using FAISS
- Persistent feature cache to speed startup (image_features.pkl)
- Simple password-protected UI and responsive layout

---

## Table of contents
- [Requirements](#requirements)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset and indexing](#dataset-and-indexing)
- [Training / Feature extraction](#training--feature-extraction)
- [API / Endpoints](#api--endpoints)
- [Deployment](#deployment)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Requirements
- Python 3.8+ (3.10/3.11 recommended)
- See `requirments.txt` for exact Python packages used in this repo.
- GPU optional: if you have CUDA and want GPU acceleration, install `faiss-gpu` instead of `faiss-cpu` and run PyTorch with CUDA.

Install dependencies (PowerShell example):

```powershell
python -m pip install -r requirments.txt
```

Note: The repository includes `requirments.txt` (note the filename spelling) — use that file name unless you rename it.

---

## Project structure
Top-level files and folders:

```
image.py                 # Main image processing utilities
Readme.md                # This file
requirments.txt          # Python dependencies (note filename)
run.bat                  # Helper script to run the app on Windows
static/
	images/                # Reference images (indexed)
	uploads/               # Uploaded query images
templates/
	index.html             # Frontend
```

---

## Installation

1. Clone the repo:

```powershell
git clone <your-repo-url>
cd Image_Search_Engine_Using_CNN
```

2. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirments.txt
```

3. Prepare folders (if they don't exist):

```powershell
mkdir static\images; mkdir static\uploads
```

---

## Usage

Run the app (there is a `run.bat` helper for Windows):

```powershell
python image.py
# or
./run.bat
```

Open your browser at http://localhost:5050 (or the address shown in the terminal).

Features in the UI:
- Upload an image to query
- Pick number of results
- Preview and inspect results

Password protection: the frontend prompts for a password. Change the default by editing `templates/index.html` (look for `CORRECT_PASSWORD`).

---

## Dataset and indexing
- Place reference images into `static/images/`. Supported formats: JPG, PNG, WebP.
- The code builds a FAISS index from image embeddings and saves features to `image_features.pkl` to avoid re-processing.
- To reindex (rebuild features), use the reindex endpoint or delete `image_features.pkl` and restart the app.

Example reindex (curl):

```powershell
curl -X POST http://localhost:5050/reindex
```

---

## Training / Feature extraction
This project does not train new models end-to-end. Instead, it extracts features from pretrained models:
- CLIP ViT-B/32 → 512-d semantic embeddings
- ResNet50 → pooled deep visual features (~4096-d in this project)
- HSV color histogram → 512-d

All vectors are concatenated and L2-normalized before building the FAISS index.

If you want to change or extend feature extractors, update `image.py` accordingly.

---

## API / Endpoints
Typical endpoints (see app code for exact routes):

- GET /            → UI page
- POST /search    → Upload/query image and return nearest neighbors
- POST /reindex   → Rebuild the feature index from images in `static/images/`

Check the Flask app source (`image.py` or `app.py`) for parameter details and response formats.

---

## Deployment

Local (Windows): use the scripts above or `run.bat`.

Production suggestions:
- Run behind a WSGI server (gunicorn or waitress)
- Use a managed service (Heroku, Railway) or containerize with Docker
- Secure the UI and set a strong password when exposing externally

---

## Development

- Add or remove images under `static/images/` and reindex.
- If you modify feature extraction or model loading, add unit tests where possible.

---

## Contributing

Contributions are welcome. Please open issues or pull requests with:
- Clear description of the change
- Small, focused commits
- If adding features that change data formats, include migration notes

---

## License
MIT License — see LICENSE file (or add one) for full terms.

---

## Acknowledgements
- OpenAI CLIP
- FAISS (Facebook AI)
- Torchvision models
- Tailwind CSS

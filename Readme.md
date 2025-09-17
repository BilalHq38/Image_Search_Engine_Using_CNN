# Image Search Engine Using CNN

A compact, self-hosted image search engine built with Flask and deep visual embeddings. It uses CLIP and ResNet for feature extraction plus HSV color histograms and FAISS for fast similarity search.


## Problem Statement

In domains like jewelery designing, fashion, and creative industries, organizations often accumulate thousands of images over time. Without an efficient search system, it becomes extremely difficult to check whether a specific design or concept already exists in the collection. Traditional keyword-based search is insufficient for visual data, as many designs lack descriptive metadata.
To address this challenge, this project provides a content-based image search engine that allows users to find visually similar images by uploading an example. By combining semantic, visual, and color-based embeddings, it enables accurate and efficient discovery of related designs.

## Key Features

- 🔍 Search by image (upload an image to find visually similar images)
- 🧠 Multi-modal embeddings: CLIP (ViT-B/32), ResNet50, and HSV color histograms
- ⚡ Fast nearest-neighbor search using FAISS
- 💾 Persistent feature cache to speed startup (`image_features.pkl`)
- 🔐 Simple password-protected UI with a responsive layout

The results are fetched from the database on the basis of similarity score

## Table of Contents

- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset & Indexing](#dataset--indexing)
- [Training / Feature Extraction](#training--feature-extraction)
- [API / Endpoints](#api--endpoints)
- [Deployment](#deployment)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Requirements

- Python 3.8+ (3.10/3.11 recommended)

See `requirments.txt` for exact Python packages used in this repo.

GPU optional: if you have CUDA and want GPU acceleration, install `faiss-gpu` instead of `faiss-cpu` and run PyTorch with CUDA.

Install dependencies (PowerShell example):

```powershell
python -m pip install -r requirments.txt
```

Note: The repository includes `requirments.txt` (note the filename spelling) — use that file name unless you rename it.

## Project Structure

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

## Installation

Clone the repo:

```powershell
git clone <your-repo-url>
cd Image_Search_Engine_Using_CNN
```

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirments.txt
```

Prepare folders (if they don't exist):

```powershell
mkdir static\images; mkdir static\uploads
```

## Usage

Run the app (there is a `run.bat` helper for Windows):

```powershell
python image.py
# or
./run.bat
```

Open your browser at http://localhost:5050

### UI Features

- Upload an image to query
- Select number of results to display
- Preview and inspect results

- 🔑 Password protection: The frontend prompts for a password. Change the default by editing `templates/index.html` (look for `CORRECT_PASSWORD`).

## Dataset and Indexing

Place reference images into `static/images/` (JPG, PNG, WebP supported).

The app builds a FAISS index from embeddings and saves features in `image_features.pkl` to avoid reprocessing.

To reindex (rebuild features), use the `/reindex` endpoint or delete `image_features.pkl` and restart the app.

Example reindex (curl):

```powershell
curl -X POST http://localhost:5050/reindex
```

## Training / Feature Extraction

This project does not train models end-to-end. Instead, it extracts features using pretrained models:

- CLIP ViT-B/32 → 512-d semantic embeddings
- ResNet50 → pooled deep visual features (~4096-d)
- HSV color histogram → 512-d

All vectors are concatenated and L2-normalized before building the FAISS index.

To extend or modify feature extractors, update `image.py`.

## API / Endpoints

- GET / → UI page
- POST /search → Upload/query image and return nearest neighbors
- POST /reindex → Rebuild feature index from images in `static/images/`

Check `image.py` for details on request/response formats.

## Deployment

Local (Windows): use `python image.py` or `run.bat`.

Production suggestions:

- Run behind a WSGI server (e.g., gunicorn or waitress)
- Deploy via Docker or a managed platform (Heroku, Railway, etc.)
- Secure the UI with a strong password before public deployment

## Development

- Add/remove images in `static/images/` and reindex
- Update feature extraction logic in `image.py` if integrating new models
- Add unit tests for custom changes where possible

## Contributing

Contributions are welcome! Please submit issues or pull requests with:

- Clear description of changes
- Small, focused commits
- Migration notes if data format changes

## License

MIT License — see LICENSE file for details.

## Acknowledgements

- OpenAI CLIP
- Facebook AI FAISS
- Torchvision models
- Tailwind CSS

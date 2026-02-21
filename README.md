<div align="center">

# ⚡ DefectScan AI
### Industrial Steel Surface Defect Detection System

[![Python](https://img.shields.io/badge/Python-3.10.19-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?style=flat-square&logo=render&logoColor=white)](https://render.com)

**A production-ready end-to-end MLOps system for multi-class steel surface defect classification with open-set rejection, REST API, and interactive web interface.**

[🌐 Live Demo](https://industrial-defect-detection-1.streamlit.app) &nbsp;·&nbsp; [📡 API Docs](https://industrial-defect-detection-sp8i.onrender.com/docs) &nbsp;·&nbsp; [🔍 Health Check](https://industrial-defect-detection-sp8i.onrender.com/health)

</div>

---

## 📌 Overview

DefectScan AI is a full-stack machine learning system built for automated quality control in industrial steel manufacturing. The system classifies surface defects across **6 defect categories** using a fine-tuned **EfficientNetB0** model, rejects irrelevant inputs via **open-set recognition**, and serves predictions through a production **FastAPI** backend — all containerized with Docker and deployed to the cloud.

> **Designed to demonstrate:** Deep learning · MLOps · REST API design · Containerization · Cloud deployment · Software engineering best practices

---

## 🏗️ System Architecture

```
Streamlit UI  →  FastAPI Backend  →  EfficientNetB0 Model
(Streamlit Cloud)    (Render)          (TensorFlow 2.20.0)
```

---

## ✨ Key Features

| Feature | Details |
|---|---|
| **Multi-class Classification** | 6 steel defect types from NEU Surface Defect Dataset |
| **Open-Set Rejection** | 7th `unknown` class rejects non-steel / irrelevant images |
| **Confidence Thresholding** | Configurable threshold (default 0.85) flags uncertain predictions |
| **Production REST API** | FastAPI with `/predict`, `/health`, structured logging, CORS |
| **Interactive Web UI** | Streamlit frontend with animated scan interface and session history |
| **Full Containerization** | Separate Dockerfiles for API and UI with Docker Compose orchestration |
| **Cloud Deployment** | Backend on Render, frontend on Streamlit Cloud |
| **Config-Driven Pipeline** | YAML-based config for reproducible training and deployment |

---

## 🧠 Model

### Architecture
```
Input (224×224×3)
    └── Data Augmentation (flip, rotation, zoom)
        └── EfficientNetB0 (ImageNet pretrained — frozen in Phase 1)
            └── GlobalAveragePooling2D
                └── BatchNormalization
                    └── Dense(256, ReLU)
                        └── Dropout(0.4)
                            └── Dense(7, Softmax)  ← 6 defects + unknown
```

### Training Strategy
- **Phase 1 — Head Training:** Backbone frozen, classifier head trained from scratch
- **Phase 2 — Fine-tuning:** Top 20% of backbone layers unfrozen, trained at reduced LR
- **Callbacks:** `ModelCheckpoint` · `EarlyStopping` · `ReduceLROnPlateau`

### Open-Set Recognition
Standard classifiers confidently misclassify out-of-distribution inputs. This system addresses that by training an explicit `unknown` class on diverse non-steel images, enabling the model to distinguish *"I recognize this defect"* from *"this isn't a steel surface at all."*

---

## 📊 Dataset

| Split | Samples | Classes |
|---|---|---|
| Training | 240 steel + non-steel images | 7 (6 defects + unknown) |
| Validation | 60 steel + non-steel images | 7 |

**Source:** [NEU Surface Defect Database — Kaggle](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)

> **⚠️ Dataset Setup**
>
> 1. Download the dataset from the Kaggle link above
> 2. Extract and place it according to the following structure:
>
> ```
> data/
> └── raw/
>     └── NEU-DET/
>         ├── train/
>         │   ├── crazing/
>         │   ├── inclusion/
>         │   ├── patches/
>         │   ├── pitted_surface/
>         │   ├── rolled-in_scale/
>         │   ├── scratches/
>         │   └── unknown/        ← add non-steel images here
>         └── validation/
>             ├── crazing/
>             ├── inclusion/
>             ├── patches/
>             ├── pitted_surface/
>             ├── rolled-in_scale/
>             ├── scratches/
>             └── unknown/        ← add non-steel images here
> ```
>
> The `unknown/` folders are not part of the original dataset — collect a set of random non-steel images and place them there to enable open-set rejection during training.

### Defect Classes

| Class | Description |
|---|---|
| `crazing` | Network of fine surface cracks |
| `inclusion` | Foreign particles embedded in the steel |
| `patches` | Irregular blotchy surface regions |
| `pitted_surface` | Small pits or craters across the surface |
| `rolled-in_scale` | Oxide scale pressed into the surface during rolling |
| `scratches` | Linear surface abrasions |
| `unknown` *(open-set)* | Non-steel / irrelevant images — rejected by the backend |

---

## 📁 Project Structure

```
Industrial-Defect-Detection/
│
├── app/                          # FastAPI application
│   ├── core/                     # App configuration & startup
│   ├── services/                 # Service layer
│   ├── dashboard.py              # Inference service logic
│   ├── main.py                   # FastAPI entry point
│   ├── schemas.py                # Pydantic request/response models
│   └── requirements.txt
│
├── config/                       # YAML training configs
├── data/
│   ├── processed/                # Preprocessed datasets
│   └── raw/NEU-DET/
│       ├── train/
│       └── validation/
│
├── entrypoint/                   # Docker entrypoint scripts
├── experiments/                  # Experiment tracking & results
├── logs/                         # Structured application logs
├── models/                       # Saved model artifacts (.keras)
├── notebooks/                    # EDA & training notebooks
├── scripts/                      # Training & utility scripts
├── src/                          # Core ML source (model, data pipeline)
├── tests/                        # Unit & integration tests
│
├── class_names.json              # Class index mapping
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.streamlit
├── .dockerignore
├── .gitignore
└── requirements.txt
```

---

## 🔌 API Reference

### `GET /health`
```bash
curl https://industrial-defect-detection-sp8i.onrender.com/health
```
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `POST /predict`
```bash
curl -X POST https://industrial-defect-detection-sp8i.onrender.com/predict \
  -F "file=@steel_sample.jpg"
```

**Defect detected:**
```json
{
  "status": "success",
  "predicted_class": "pitted_surface",
  "confidence": 0.9731
}
```

**Non-steel image rejected:**
```json
{
  "status": "invalid_input",
  "message": "Please upload a steel surface defect image."
}
```

**Low-confidence prediction:**
```json
{
  "status": "uncertain",
  "predicted_class": "scratches",
  "confidence": 0.6142
}
```

Full interactive docs available at [`/docs`](https://industrial-defect-detection-sp8i.onrender.com/docs).

---

## ⚙️ Configuration (`config/local.yaml`)

```yaml
project:
  name: industrial-defect-detection
  seed: 42

data:
  train_dir: data/raw/NEU-DET/train
  val_dir: data/raw/NEU-DET/validation

model:
  backbone: efficientnet_b0
  img_size: [224, 224]
  num_classes: 7
  model_path: models/defect_detector_finetuned_v2.keras

training:
  batch_size: 32
  epochs: 10
  learning_rate: 0.001
  fine_tune_learning_rate: 0.00001

inference:
  confidence_threshold: 0.95
```

---

## 🚀 Getting Started

### Prerequisites
- Docker & Docker Compose, **or** Python 3.10.19+

### Option 1 — Docker Compose *(recommended)*

```bash
git clone https://github.com/PranjalSabhaya/Industrial-Defect-Detection.git
cd Industrial-Defect-Detection
docker-compose up --build
```

| Service | URL |
|---|---|
| FastAPI Backend | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |

### Option 2 — Local Development

**Backend:**
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend** *(separate terminal):*
```bash
pip install streamlit requests
streamlit run app.py
```

---

## 🐳 Docker

Each service has its own optimized Dockerfile:

```yaml
# docker-compose.yml
services:
  api:
    build:
      dockerfile: Dockerfile.api
    ports: ["8000:8000"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  streamlit:
    build:
      dockerfile: Dockerfile.streamlit
    ports: ["8501:8501"]
    depends_on: [api]
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Model** | TensorFlow 2.20.0 · EfficientNetB0 · Keras |
| **Backend** | FastAPI · Uvicorn · Pydantic · Python 3.10.19 |
| **Frontend** | Streamlit |
| **Containerization** | Docker · Docker Compose |
| **Deployment** | Render *(API)* · Streamlit Cloud *(UI)* |
| **Config Management** | YAML |
| **Logging** | Python `logging` · Structured logs |

---

---

## 👤 Author

**Pranjal Sabhaya**

[![GitHub](https://img.shields.io/badge/GitHub-PranjalSabhaya-181717?style=flat-square&logo=github)](https://github.com/PranjalSabhaya)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/pranjal-sabhaya-505391286)

---


<div align="center">
<sub>Built with TensorFlow · FastAPI · Streamlit · Docker</sub>
</div>

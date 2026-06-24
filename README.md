# Adhera Server 

Adhera Server is a multimodal AI-powered backend system designed to process different types of medical and behavioral data and provide intelligent predictions through a unified API interface.

The server integrates multiple machine learning and deep learning models to analyze data from several modalities including MRI scans, EEG signals, facial expressions, eye tracking, and questionnaire responses.

## Features

- MRI-based prediction pipeline
- EEG signal analysis and classification
- Facial expression and engagement analysis from video
- Eye tracking feature prediction
- Questionnaire-based prediction
- REST API built with FastAPI
- GPU support for accelerated inference
- Automatic preprocessing for each input type
- Health monitoring endpoint

---

## Supported Modalities

### MRI Analysis
Processes MRI scans through:

- Brain masking
- Image normalization
- ROI extraction using AAL Atlas
- Transformer-based prediction model

### EEG Analysis
Processes EEG signals through:

- Signal filtering (1–50 Hz Bandpass)
- Window segmentation
- Z-score normalization
- CNN-based classification

### Facial Analysis
Processes video input through:

- Facial feature extraction
- Sequential frame analysis using LSTM
- Engagement prediction

### Eye Tracking Analysis
Processes eye tracking engineered features through:

- Missing value handling
- Feature scaling
- Transformer-based prediction

### Questionnaire Analysis
Processes questionnaire responses using a trained machine learning model.

---

## API Endpoints

| Method | Endpoint | Description |
|----------|-----------|-------------|
| GET | `/` | Server status |
| GET | `/health` | Check model loading status |
| POST | `/predict/mri` | MRI prediction |
| POST | `/predict/questionnaire` | Questionnaire prediction |
| POST | `/predict/eeg` | EEG prediction |
| POST | `/predict/facial` | Facial prediction |
| POST | `/predict/eye-tracking` | Eye tracking prediction |

---

## Project Structure

```bash
Adhera Server/
│
├── Models/
│   ├── MRI Models
│   ├── EEG Models
│   ├── Facial Models
│   ├── EyeTrackingAssets/
│
├── preprocessing/
│
├── utils/
│
├── main.py
│
└── README.md
```

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Server

To run the server, open the terminal inside the project directory and run:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

This command starts the server locally with auto-reload enabled, so changes will be reflected automatically during development.

---

## Access API Documentation

After starting the server, open:

Swagger UI:

```bash
http://localhost:8000/docs
```

ReDoc:

```bash
http://localhost:8000/redoc
```

---

## Health Check Example

Request:

```bash
GET /health
```

Response:

```json
{
    "mri_model": "loaded",
    "questionnaire_model": "loaded",
    "eeg_model": "loaded",
    "facial_model": "loaded",
    "eye_tracking_model": "loaded"
}
```

---

## Technologies Used

- FastAPI
- PyTorch
- TensorFlow
- NumPy
- Pandas
- Nilearn
- Nibabel
- Scikit-learn
- Joblib

---

## Notes

- GPU acceleration is automatically used when available.
- Each prediction pipeline includes preprocessing before inference.
- MRI scans should be provided as NIfTI files (`.nii` or `.nii.gz`).
- EEG input should contain 19 channels.
- Facial prediction expects video input.
- Eye tracking prediction expects engineered feature vectors.


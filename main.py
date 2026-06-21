import os
import io
import gzip
import torch
import torch.nn as nn
import pickle
import joblib
import numpy as np
import nibabel as nib
import tensorflow as tf
from scipy.signal import butter, filtfilt
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List

from nilearn.masking import compute_brain_mask
from nilearn.image import math_img, resample_img, smooth_img
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker

from utils.gpu import get_device
from utils.facial import extract_features_from_video

app = FastAPI(title="Adhera Server V1.0")
device = get_device()

# ===================== Model Architectures =====================

class FTTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=4, n_layers=3, dropout=0.15):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.dropout(x)
        return self.classifier(x)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        return hn[-1]


class CombinedFacialModel(nn.Module):
    def __init__(self, model_full, model_face, mlp_classifier):
        super(CombinedFacialModel, self).__init__()
        self.model_full = model_full
        self.model_face = model_face
        self.mlp_classifier = mlp_classifier

    def forward(self, x_full, x_face):
        full_out = self.model_full(x_full)
        face_out = self.model_face(x_face)
        combined = torch.cat((full_out, face_out), dim=1)
        logits = self.mlp_classifier(combined)
        return logits


# ===================== Load AAL Atlas =====================
print("Loading AAL Atlas...")
aal_dataset = datasets.fetch_atlas_aal(version='SPM12')

# ===================== Model Loading =====================

# 1. Questionnaire Model
print("\n=== Loading Questionnaire Model ===")
try:
    with open('E:\\Adhera Server\\Models\\best_xgb_TRAQ10.pkl', 'rb') as f:
        questionnaire_model = pickle.load(f)
    print("[OK] Questionnaire Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Error loading Questionnaire: {e}")
    questionnaire_model = None

# 2. EEG Model (best_cnn.keras)
print("\n=== Loading EEG Model ===")
try:
    eeg_model = tf.keras.models.load_model('E:\\Adhera Server\\Models\\best_cnn.keras')
    print("[OK] EEG Model (best_cnn.keras) loaded successfully.")
except Exception as e:
    print(f"[ERROR] Error loading EEG Model: {e}")
    eeg_model = None

# 3. MRI Model & Scaler
print("\n=== Loading MRI Model ===")
try:
    with open('E:\\Adhera Server\\preprocessing\\scaler.pkl', 'rb') as f:
        mri_scaler = joblib.load(f)

    mri_model = FTTransformer(input_dim=117).to(device)
    state_dict = torch.load('E:\\Adhera Server\\Models\\best_ft_transformer.pt', map_location=device)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        mri_model.load_state_dict(state_dict['model_state_dict'])
    else:
        mri_model.load_state_dict(state_dict)
    mri_model.eval()
    mri_threshold = 0.51
    print("[OK] MRI Model & Scaler loaded successfully.")
except Exception as e:
    print(f"[ERROR] Error loading MRI: {e}")
    mri_model = None
    mri_scaler = None

# 4. Facial Model (best_combined_model_lstm_binary)
print("\n=== Loading Facial Model ===")
facial_model = None
try:
    # Initialize component models
    hidden_size = 1024
    feature_dim = 1280
    num_layers = 1
    dropout_rate = 0.5
    num_classes = 4
    combined_feature_size = hidden_size * 2

    model_full = LSTMClassifier(feature_dim, hidden_size, num_layers, dropout_rate).to(device)
    model_face = LSTMClassifier(feature_dim, hidden_size, num_layers, dropout_rate).to(device)
    mlp_classifier = nn.Sequential(
        nn.Linear(combined_feature_size, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    ).to(device)

    facial_model = CombinedFacialModel(model_full, model_face, mlp_classifier).to(device)

    # Load checkpoint - try multiple paths
    checkpoint_paths = [
        'E:\\Adhera Server\\Models\\best_combined_model_lstm_binary\\best_combined_model_lstm.pt',
        'E:\\Adhera Server\\Models\\best_combined_model_lstm.pt',
        'E:\\Adhera Server\\Models\\best_combined_model_lstm_binary'
    ]

    checkpoint = None
    for path in checkpoint_paths:
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            print(f"[OK] Loaded checkpoint from {path}")
            break
        except Exception as e:
            continue

    if checkpoint is None:
        raise FileNotFoundError(f"Could not find facial model checkpoint in any of: {checkpoint_paths}")

    facial_model.model_full.load_state_dict(checkpoint['model_full_state_dict'])
    facial_model.model_face.load_state_dict(checkpoint['model_face_state_dict'])
    facial_model.mlp_classifier.load_state_dict(checkpoint['mlp_classifier_state_dict'])
    facial_model.eval()
    print("[OK] Facial Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Error loading Facial Model: {e}")
    facial_model = None

# ===================== EEG Preprocessing Functions =====================

def bandpass_filter(signal, fs=128, lowcut=1.0, highcut=50.0, order=4):
    """Apply Butterworth bandpass filter to signal"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def process_eeg_signal_complete(content):
    """
    Complete EEG preprocessing:
    1. Parse CSV data (handles multiple encodings)
    2. Apply bandpass filter (1-50 Hz)
    3. Segment into windows (256 samples / 2 seconds)
    4. Z-score normalize per channel
    """
    try:
        # Try multiple encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'ascii']
        text_content = None

        for encoding in encodings:
            try:
                text_content = content.decode(encoding)
                break
            except:
                continue

        if text_content is None:
            raise ValueError("Could not decode file with any standard encoding")

        # Parse CSV content
        lines = text_content.strip().split('\n')
        data_rows = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to parse as comma-separated or space-separated values
            try:
                if ',' in line:
                    values = [float(x.strip()) for x in line.split(',') if x.strip()]
                else:
                    values = [float(x.strip()) for x in line.split() if x.strip()]

                if values:
                    data_rows.append(values)
            except ValueError:
                # Skip lines that can't be parsed
                continue

        if not data_rows:
            raise ValueError("No valid numeric data in EEG file")

        data = np.array(data_rows)

        # Expected: (n_samples, 19_channels)
        if data.shape[1] != 19:
            raise ValueError(f"Expected 19 channels, got {data.shape[1]}. Please provide CSV with 19 EEG channels: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, Fz, Cz, Pz")

        # Apply bandpass filter to each channel
        filtered_data = np.zeros_like(data)
        for ch in range(19):
            filtered_data[:, ch] = bandpass_filter(data[:, ch], fs=128, lowcut=1.0, highcut=50.0)

        # Segment into 256-sample windows (2 seconds at 128 Hz)
        window_size = 256
        windows = []
        for start in range(0, len(filtered_data) - window_size + 1, window_size):
            window = filtered_data[start:start + window_size]
            windows.append(window)

        if not windows:
            raise ValueError(f"Not enough samples to create windows. Need at least {window_size} samples, got {len(filtered_data)}")

        # Stack windows and normalize per channel
        X = np.array(windows)  # Shape: (n_windows, 256, 19)

        # Z-score normalize per channel across all windows and samples
        for ch in range(19):
            mean = X[:, :, ch].mean()
            std = X[:, :, ch].std() + 1e-8
            X[:, :, ch] = (X[:, :, ch] - mean) / std

        return X

    except Exception as e:
        raise Exception(f"EEG Processing Error: {e}")


def process_mri_stream(image_bytes):
    """MRI preprocessing pipeline"""
    fh = nib.FileHolder(fileobj=io.BytesIO(image_bytes))
    raw_img = nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})
    mask_img = compute_brain_mask(raw_img)
    img_stripped = math_img('img * mask', img=raw_img, mask=mask_img)
    img_resampled = resample_img(img_stripped, target_affine=np.diag([2, 2, 2]))
    img_normalized = math_img('(img - np.mean(img)) / (np.std(img) + 1e-8)', img=img_resampled)
    final_img = smooth_img(img_normalized, fwhm=6)

    masker = NiftiLabelsMasker(labels_img=aal_dataset.maps, standardize=False)
    roi_values = masker.fit_transform(final_img).flatten()
    site_feature = np.array([0])
    full_features = np.hstack([roi_values, site_feature]).reshape(1, -1)
    return full_features

# ===================== API Data Models =====================

class QuestionnaireData(BaseModel):
    features: List[float]

# ===================== Endpoints =====================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "version": "1.0",
        "available_endpoints": [
            "/predict/mri",
            "/predict/questionnaire",
            "/predict/eeg",
            "/predict/facial"
        ]
    }


@app.get("/health")
async def health():
    """Check if all models are loaded"""
    return {
        "mri_model": "loaded" if mri_model is not None else "not_loaded",
        "questionnaire_model": "loaded" if questionnaire_model is not None else "not_loaded",
        "eeg_model": "loaded" if eeg_model is not None else "not_loaded",
        "facial_model": "loaded" if facial_model is not None else "not_loaded"
    }


@app.post("/predict/mri")
async def predict_mri(file: UploadFile = File(...)):
    """Predict ADHD from MRI scan"""
    try:
        if mri_model is None:
            return {"status": "error", "message": "MRI model not loaded"}

        content = await file.read()
        if file.filename.endswith('.gz'):
            with gzip.GzipFile(fileobj=io.BytesIO(content)) as f:
                content = f.read()

        features = process_mri_stream(content)
        if mri_scaler:
            features = mri_scaler.transform(features)

        input_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = mri_model(input_tensor)
            prob = torch.sigmoid(logits).item()
            prediction = 1 if prob >= mri_threshold else 0
            probability_percent = round(prob * 100, 2)

        return {
            "status": "success",
            "prediction": prediction,
            "probability": probability_percent
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/predict/questionnaire")
async def predict_questionnaire(data: QuestionnaireData):
    """Predict ADHD from questionnaire responses"""
    try:
        if questionnaire_model is None:
            return {"status": "error", "message": "Questionnaire model not loaded"}

        input_array = np.array(data.features).reshape(1, -1)
        prediction = questionnaire_model.predict(input_array)
        return {"status": "success", "prediction": int(prediction[0])}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/predict/eeg")
async def predict_eeg(file: UploadFile = File(...)):
    """Predict ADHD from EEG signal"""
    try:
        if eeg_model is None:
            return {"status": "error", "message": "EEG model not loaded"}

        content = await file.read()

        # Preprocessing
        X = process_eeg_signal_complete(content)

        # Prediction
        predictions = eeg_model.predict(X, verbose=0)

        # Convert numpy → safe python types
        predictions = np.array(predictions)

        avg_prob = float(np.mean(predictions))   # 👈 مهم جدًا
        prediction = int(avg_prob >= 0.5)        # 👈 ensure Python int
        probability_percent = float(round(avg_prob * 100, 2))  # 👈 safe float

        return {
            "status": "success",
            "prediction": prediction,
            "probability": probability_percent,
            "message": f"Analyzed {len(X)} EEG windows"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/predict/facial")
async def predict_facial(file: UploadFile = File(...)):
    """Predict engagement level from facial expression video"""
    try:
        if facial_model is None:
            return {"status": "error", "message": "Facial model not loaded"}

        # Save video temporarily
        video_path = f"/tmp/{file.filename}"
        os.makedirs('/tmp', exist_ok=True)

        content = await file.read()
        with open(video_path, 'wb') as f:
            f.write(content)

        # Extract features from video
        full_features, face_features = extract_features_from_video(video_path, device)

        # Convert to tensors
        full_tensor = torch.tensor(full_features, dtype=torch.float32).unsqueeze(0).to(device)
        face_tensor = torch.tensor(face_features, dtype=torch.float32).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            logits = facial_model(full_tensor, face_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = round(probs[0, pred_class].item() * 100, 2)

        # Clean up
        os.remove(video_path)

        engagement_levels = ["Very Low", "Low", "High", "Very High"]

        return {
            "status": "success",
            "engagement_level": pred_class,
            "engagement_label": engagement_levels[pred_class],
            "confidence": confidence
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

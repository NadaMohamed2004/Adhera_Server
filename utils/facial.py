import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


_feature_extractor = None


def extract_frames_from_video(video_path, num_frames=20):
    """
    Extract evenly spaced readable frames from a video.

    Args:
        video_path (str): Path to video file
        num_frames (int): Number of frames to extract

    Returns:
        list: List of frames as numpy arrays (RGB format)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Could not read video: {video_path}")

    frame_idxs = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in frame_idxs:
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        except Exception as frame_error:
            print(f"[FACIAL] Skipping unreadable frame {idx}: {frame_error}")
            continue

    cap.release()

    if not frames:
        raise ValueError(f"No valid frames extracted from video: {video_path}")

    print(f"[FACIAL] Extracted {len(frames)} frames")

    return frames


def crop_faces(frames):
    """Crop the largest detected face per frame; fall back to the full frame."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cropped = []

    for frame in frames:
        try:
            if frame is None or frame.size == 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                height, width = frame.shape[:2]
                x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
                x = max(0, int(x))
                y = max(0, int(y))
                w = min(int(w), width - x)
                h = min(int(h), height - y)
                if w > 0 and h > 0:
                    face = frame[y:y + h, x:x + w]
                    if face.size > 0:
                        face = cv2.resize(face, (width, height))
                        cropped.append(face)
                        continue
            cropped.append(frame)
        except Exception as face_error:
            print(f"[FACIAL] Face crop failed, using original frame: {face_error}")
            cropped.append(frame)

    return cropped


def get_facial_preprocessing():
    """Returns preprocessing transforms for EfficientNetV2-S."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_feature_extractor(device):
    """Load EfficientNetV2-S once and reuse it across requests."""
    global _feature_extractor
    if _feature_extractor is None:
        model = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT
        )
        model.classifier = nn.Identity()
        model = model.to(device)
        model.eval()
        _feature_extractor = model
    return _feature_extractor


def extract_features_from_video(video_path, device, num_frames=20):
    """
    Extract EfficientNetV2-S features from video frames.

    Args:
        video_path (str): Path to video file
        device (torch.device): Device to run model on
        num_frames (int): Number of frames to extract

    Returns:
        tuple: (full_features, face_features) - shape (valid_frames, 1280) each
    """
    full_batch = None
    face_batch = None
    try:
        model = get_feature_extractor(device)
        preprocess = get_facial_preprocessing()

        frames = extract_frames_from_video(video_path, num_frames=num_frames)
        face_frames = crop_faces(frames)

        try:
            full_tensors = [preprocess(Image.fromarray(frame)) for frame in frames]
            face_tensors = [preprocess(Image.fromarray(frame)) for frame in face_frames]
        except Exception as preprocess_error:
            raise ValueError(f"Could not preprocess video frames: {preprocess_error}")

        if not full_tensors or not face_tensors:
            raise ValueError("No valid tensors created from video frames")

        full_batch = torch.stack(full_tensors).to(device)
        face_batch = torch.stack(face_tensors).to(device)

        with torch.no_grad():
            full_features = model(full_batch).cpu().numpy()
            face_features = model(face_batch).cpu().numpy()

        print(f"[FACIAL] Feature extraction complete")
        return full_features, face_features
    finally:
        if full_batch is not None:
            del full_batch
        if face_batch is not None:
            del face_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

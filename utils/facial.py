import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

def extract_frames_from_video(video_path, num_frames=60):
    """
    Extract evenly spaced frames from a video.

    Args:
        video_path (str): Path to video file
        num_frames (int): Number of frames to extract

    Returns:
        list: List of frames as numpy arrays (RGB format)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        raise ValueError(f"Could not read video: {video_path}")

    frame_idxs = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) < num_frames * 0.8:
        raise ValueError(f"Could only extract {len(frames)} frames out of {num_frames}")

    return frames


def get_facial_preprocessing():
    """Returns preprocessing transforms for EfficientNetV2-L"""
    return transforms.Compose([
        transforms.Resize(600),
        transforms.CenterCrop(600),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def extract_features_from_video(video_path, device, num_frames=60):
    """
    Extract EfficientNetV2-L features from video frames.

    Args:
        video_path (str): Path to video file
        device (torch.device): Device to run model on
        num_frames (int): Number of frames to extract

    Returns:
        tuple: (full_features, face_features) - shape (num_frames, 1280) each
    """
    # Load pretrained EfficientNetV2-L
    model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
    model.classifier = nn.Identity()
    model = model.to(device)
    model.eval()

    preprocess = get_facial_preprocessing()

    # Extract frames
    frames = extract_frames_from_video(video_path, num_frames=num_frames)

    # Preprocess frames
    pil_frames = [Image.fromarray(frame) for frame in frames]
    frame_tensors = [preprocess(img) for img in pil_frames]
    batch_tensor = torch.stack(frame_tensors).to(device)

    # Extract features
    with torch.no_grad():
        features = model(batch_tensor)

    # Return both full and face features as the same (they're both EfficientNet outputs)
    # In production, you'd have separate face detector to crop faces, but for now using full frames
    features_np = features.cpu().numpy()

    # Return as (full_features, face_features) - both using full frame features
    # The model architecture expects both streams
    return features_np, features_np

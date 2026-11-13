"""Extract fixed-length pose and hand landmark sequences from videos."""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

FEATURE_SIZE = 258


def create_holistic():
    """Create the MediaPipe Holistic tracker used by training and inference."""
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _landmarks_to_features(results) -> np.ndarray:
    pose = np.zeros((33, 4), dtype=np.float32)
    if results.pose_landmarks:
        pose = np.asarray(
            [
                (point.x, point.y, point.z, point.visibility)
                for point in results.pose_landmarks.landmark
            ],
            dtype=np.float32,
        )

    left_hand = np.zeros((21, 3), dtype=np.float32)
    if results.left_hand_landmarks:
        left_hand = np.asarray(
            [(point.x, point.y, point.z) for point in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        )

    right_hand = np.zeros((21, 3), dtype=np.float32)
    if results.right_hand_landmarks:
        right_hand = np.asarray(
            [(point.x, point.y, point.z) for point in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        )

    return np.concatenate((pose.ravel(), left_hand.ravel(), right_hand.ravel()))


def extract_video_features(
    video_path: str | Path, holistic, sequence_length: int = 30
) -> np.ndarray:
    """Sample a video uniformly and return (sequence_length, FEATURE_SIZE)."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError(f"Video contains no readable frames: {video_path}")

        frame_indices = np.linspace(0, frame_count - 1, sequence_length, dtype=int)
        sequence = np.zeros((sequence_length, FEATURE_SIZE), dtype=np.float32)
        for index, frame_index in enumerate(frame_indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = capture.read()
            if not success:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)
            sequence[index] = _landmarks_to_features(results)
        return sequence
    finally:
        capture.release()

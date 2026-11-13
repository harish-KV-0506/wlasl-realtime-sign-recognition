"""Run real-time sign prediction from a webcam."""

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from sign_language.landmarks import FEATURE_SIZE, create_holistic, _landmarks_to_features
from sign_language.model import SignSequenceClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"Model checkpoint not found: {args.model}")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    labels = checkpoint["labels"]
    sequence_length = int(checkpoint["sequence_length"])
    input_size = int(checkpoint["input_size"])
    if input_size != FEATURE_SIZE or not labels:
        parser.error("Checkpoint has an unsupported feature layout or no labels.")

    model = SignSequenceClassifier(
        num_classes=len(labels),
        input_size=input_size,
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["num_layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        parser.error(f"Could not open webcam {args.camera}.")

    sequence = deque(maxlen=sequence_length)
    try:
        with create_holistic() as holistic:
            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("Could not read a frame from the webcam.")

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb_frame)
                sequence.append(_landmarks_to_features(results))

                text = "Show a sign..."
                if len(sequence) == sequence_length:
                    batch = torch.tensor(
                        np.asarray(sequence), dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        probabilities = model(batch).softmax(dim=1)[0]
                    confidence, predicted = probabilities.max(dim=0)
                    score = float(confidence.item())
                    text = (
                        f"{labels[int(predicted.item())]}  {score:.0%}"
                        if score >= args.threshold
                        else "Uncertain"
                    )

                cv2.putText(
                    frame,
                    text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Real-Time Sign Language Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

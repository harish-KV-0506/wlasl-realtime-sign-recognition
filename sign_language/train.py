"""Train and evaluate the landmark sequence classifier on WLASL."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sign_language.landmarks import FEATURE_SIZE, create_holistic, extract_video_features
from sign_language.model import SignSequenceClassifier

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def _find_video(videos_dir: Path, video_id: str) -> Path | None:
    for extension in VIDEO_EXTENSIONS:
        candidate = videos_dir / f"{video_id}{extension}"
        if candidate.is_file():
            return candidate
    return None


def _read_records(annotations_path: Path, videos_dir: Path):
    with annotations_path.open(encoding="utf-8") as annotation_file:
        entries = json.load(annotation_file)
    if not isinstance(entries, list):
        raise ValueError("WLASL annotations must contain a JSON list.")

    records = []
    gloss_counts = Counter()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("instances"), list):
            continue
        gloss = entry.get("gloss")
        if not isinstance(gloss, str):
            continue
        for instance in entry["instances"]:
            if not isinstance(instance, dict):
                continue
            gloss_counts[gloss] += 1
            video_id = instance.get("video_id")
            split = instance.get("split")
            if not isinstance(video_id, str) or split not in ("train", "val", "test"):
                continue
            video_path = _find_video(videos_dir, video_id)
            if video_path is not None:
                records.append(
                    {
                        "gloss": gloss,
                        "video_id": video_id,
                        "split": split,
                        "path": video_path,
                    }
                )
    return records, gloss_counts


def _cache_path(cache_dir: Path, video_id: str, sequence_length: int) -> Path:
    safe_id = "".join(char for char in video_id if char.isalnum() or char in "-_")
    return cache_dir / f"{safe_id}_{sequence_length}_v1.npy"


def _load_features(record, cache_dir: Path, holistic, sequence_length: int):
    path = _cache_path(cache_dir, record["video_id"], sequence_length)
    if path.is_file():
        return np.load(path)
    features = extract_video_features(record["path"], holistic, sequence_length)
    np.save(path, features)
    return features


def _evaluate(model, loader, device):
    if len(loader.dataset) == 0:
        return float("nan")
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for sequences, labels in loader:
            logits = model(sequences.to(device))
            correct += (logits.argmax(dim=1).cpu() == labels).sum().item()
            total += labels.numel()
    return correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/wlasl100.pt"))
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_classes < 2 or args.sequence_length < 1 or args.epochs < 1:
        parser.error("--num-classes must be >= 2; sequence length and epochs must be positive.")
    if not args.annotations.is_file():
        parser.error(f"Annotation file not found: {args.annotations}")
    if not args.videos.is_dir():
        parser.error(f"Videos directory not found: {args.videos}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records, frequencies = _read_records(args.annotations, args.videos)
    if not records:
        parser.error("No annotated video files were found; check the videos directory.")

    top_glosses = [
        gloss for gloss, _ in sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    ][: args.num_classes]
    train_counts = Counter(
        record["gloss"]
        for record in records
        if record["split"] == "train" and record["gloss"] in top_glosses
    )
    labels = sorted(gloss for gloss in top_glosses if train_counts[gloss] >= 2)
    if len(labels) < 2:
        parser.error(
            "Fewer than two selected signs have two available training videos. "
            "Obtain more WLASL videos or use a smaller --num-classes value."
        )
    label_to_index = {label: index for index, label in enumerate(labels)}
    records = [record for record in records if record["gloss"] in label_to_index]

    cache_dir = args.videos.parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    features_by_split = {"train": [], "val": [], "test": []}
    labels_by_split = {"train": [], "val": [], "test": []}
    skipped = 0
    with create_holistic() as holistic:
        for record in records:
            try:
                features = _load_features(
                    record, cache_dir, holistic, args.sequence_length
                )
            except (OSError, ValueError, cv2.error) as error:
                print(f"Skipping {record['video_id']}: {error}")
                skipped += 1
                continue
            features_by_split[record["split"]].append(features)
            labels_by_split[record["split"]].append(label_to_index[record["gloss"]])

    if len(features_by_split["train"]) < 2:
        parser.error("Not enough readable training videos were extracted.")
    print(
        f"Classes: {len(labels)} | "
        f"videos: train={len(features_by_split['train'])}, "
        f"val={len(features_by_split['val'])}, test={len(features_by_split['test'])} "
        f"| skipped={skipped} | device={device}"
    )

    def make_loader(split, shuffle=False):
        if not features_by_split[split]:
            empty_sequences = torch.empty(
                (0, args.sequence_length, FEATURE_SIZE), dtype=torch.float32
            )
            empty_labels = torch.empty((0,), dtype=torch.long)
            return DataLoader(
                TensorDataset(empty_sequences, empty_labels),
                batch_size=args.batch_size,
            )
        sequences = torch.tensor(np.asarray(features_by_split[split]), dtype=torch.float32)
        targets = torch.tensor(labels_by_split[split], dtype=torch.long)
        return DataLoader(
            TensorDataset(sequences, targets),
            batch_size=args.batch_size,
            shuffle=shuffle,
        )

    train_loader = make_loader("train", shuffle=True)
    val_loader = make_loader("val")
    test_loader = make_loader("test")
    model = SignSequenceClassifier(num_classes=len(labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    best_val_accuracy = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(sequences), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * targets.size(0)
        val_accuracy = _evaluate(model, val_loader, device)
        if np.isnan(val_accuracy) or val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        val_text = "n/a" if np.isnan(val_accuracy) else f"{val_accuracy:.3f}"
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"loss={running_loss / len(train_loader.dataset):.4f} "
            f"val_accuracy={val_text}"
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    model.load_state_dict(best_state)
    test_accuracy = _evaluate(model, test_loader, device)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": labels,
            "input_size": FEATURE_SIZE,
            "sequence_length": args.sequence_length,
            "hidden_size": 128,
            "num_layers": 2,
        },
        args.output,
    )
    test_text = "n/a" if np.isnan(test_accuracy) else f"{test_accuracy:.3f}"
    print(f"Test accuracy: {test_text}")
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()

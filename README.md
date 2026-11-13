# Real-Time Sign Language Recognition

A webcam sign-recognition project using MediaPipe Holistic landmarks and a
PyTorch LSTM sequence classifier, designed for the WLASL100 subset.

Project page: https://harish-KV-0506.github.io/wlasl-realtime-sign-recognition/

## Dataset access

WLASL is not bundled with this repository. Before obtaining or using it, read
and follow the dataset authors' Computational Use of Data Agreement (C-UDA):
[WLASL dataset repository](https://github.com/dxli94/WLASL). The authors'
repository documents the annotation files, video retrieval, and the process
for requesting missing or preprocessed videos. Some original video URLs may no
longer work.

After obtaining the data, arrange it like this:

```text
data/
  WLASL_v0.3.json
  videos/
    <video_id>.mp4
```

Do not upload dataset videos, annotations, or trained weights unless their
terms explicitly allow redistribution.

## Setup

Use Python 3.10 or 3.11 on Windows, macOS, or Linux:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train

Feature extraction uses MediaPipe Holistic on each video. Training caches
landmark sequences locally (under `data/.cache/`) and writes a checkpoint
locally (under `models/`):

```bash
python -m sign_language.train --annotations data/WLASL_v0.3.json --videos data/videos --output models/wlasl100.pt --num-classes 100
```

Only samples whose video files are present are used. Training reports how
many videos were skipped and requires at least two available examples per
selected sign. Use `--num-classes` to run a smaller experiment when fewer
classes have usable videos.

## Live webcam demo

Train a checkpoint first, then run:

```bash
python -m sign_language.webcam --model models/wlasl100.pt --camera 0
```

Press `q` to quit. The prediction is shown only when its confidence meets the
configured threshold; the default is 0.65. This is a research/demo project, not
a certified translation or accessibility service. Recognition quality
depends on the data, signer, camera, lighting, and training configuration.

## Project layout

```text
sign_language/
  __init__.py
  landmarks.py     # MediaPipe Holistic feature extraction
  model.py         # LSTM sequence classifier
  train.py         # WLASL100 training and evaluation
  webcam.py        # Real-time webcam inference
```

## Notes

- The model input is 258 values per frame: 33 pose landmarks (x, y, z,
  visibility) and 21 landmarks for each hand (x, y, z).
- Training uses the WLASL-provided train/validation/test splits.
- Dataset files, cached features, model checkpoints, and recordings are
  intentionally excluded from Git.
- If you use WLASL, cite the dataset authors and comply with the agreement.

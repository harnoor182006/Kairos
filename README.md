# Kairos

<div align="center">

**Action-aware video understanding for EPIC-KITCHENS-100 and FPVSum**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20%7C%20MPS%20%7C%20CPU-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Track](https://img.shields.io/badge/Samsung%20PRISM-XR%20Hackathon-Track%204-1428A0)](https://www.samsungprism.com/)

Kairos converts first-person videos into structured predictions: a **video label**, a set of **temporal actions**, and **highlight segments** ready for local validation and submission.

</div>

---

## Highlights

- **Two dataset-aware routes**: cooking actions for EPIC-KITCHENS-100 and sporting labels plus highlights for FPVSum
- **AVION + EPIC pipeline**: extracts ViT-B features, marginalizes joint actions, and decodes verb/noun predictions
- **FPV highlight pipeline**: motion, audio, semantic, temporal, and novelty signals with budget-aware selection
- **Offline-first inference**: no model download is required during normal prediction
- **Disk-conscious batching**: streams one dataset source at a time and validates before publishing
- **Deterministic fallback**: remains usable when optional models or checkpoints are unavailable

> [!IMPORTANT]
> Kairos is an engineering prototype, not a benchmark-certified submission system. AVION, FPV model checkpoints, calibration, and real dataset accuracy still require validation on the complete scoring data.

## How it works

```text
Video
  |
  +--> Probe duration and decode frames
  |
  +--> Dataset-aware routing
  |      |
  |      +--> EPIC-KITCHENS: AVION cache -> action decoding -> highlight selection
  |      |
  |      +--> FPVSum: motion/audio/semantic features -> temporal selection
  |
  +--> Canonical labels and timestamp sanitization
  |
  +--> Validated JSON prediction
```

| Component | Responsibility |
|---|---|
| `code/run_video.py` | Single-video entry point |
| `code/pipeline.py` | Decoding, routing, and branch orchestration |
| `code/avion_epic.py` | AVION checkpoint integration |
| `code/epic_actions.py` | EPIC verb/noun decoding and temporal filtering |
| `code/fpv_pipeline.py` | FPVSum feature and highlight pipeline |
| `code/kts.py` | Feature-based temporal segmentation |
| `batch.py` | 187-video manifest batch inference |
| `validate.py` | Submission coverage and schema validation |
| `tools/stream_manifest.py` | Disk-conscious streaming manifest runner |

## Quick start

### Requirements

- Python 3.10 or newer
- `ffmpeg` and `ffprobe` on `PATH`
- Optional model dependencies for GPU/model-backed inference

### Linux or macOS

```bash
python -m pip install -r requirements.txt
chmod +x run.sh
export TEAM_ID_STR="YOUR_TEAM_ID"
./run.sh input.mp4 output.json
```

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python code/run_video.py input.mp4 output.json
```

Set `TEAM_ID_STR` to the same non-empty team ID for every submission.

## Optional model setup

Base inference remains lightweight. Install the model stack separately after choosing a compatible PyTorch/CUDA build:

```bash
python -m pip install -r requirements-models.txt
python tools/download_models.py
python tools/download_avion.py
```

```text
weights/
├── siglip/                         # Optional local SigLIP checkpoint
├── videomae_kinetics/              # Optional local VideoMAE directory
├── avion/                          # Verified AVION EK100 checkpoint
└── epic_cache/<video_id>.npz       # Cached EPIC window logits
```

Model inference is offline. Normal setup does not download models unless a setup or download tool is explicitly run.

## Batch inference

The bundled manifest contains 138 EPIC-KITCHENS-100 videos and 49 FPVSum videos.

```bash
python batch.py path/to/videos outputs --team-id YOUR_TEAM_ID
python validate.py --dir outputs/
```

Missing or invalid videos fail batch inference by default. `--fill-missing` is only a fallback diagnostic mode and is not appropriate for a submission.

For low-disk full runs:

```bash
python tools/stream_manifest.py manifest.csv \
  --work-dir ./work \
  --output-dir outputs \
  --team-id YOUR_TEAM_ID
```

The streaming runner downloads and processes one source at a time, optionally prefetching only the next source. It publishes a JSON only after local validation, then cleans up its runner-owned source directory.

## AVION cache extraction

For an EPIC-KITCHENS video, an existing cache is reused; otherwise Kairos extracts it automatically when routed to cooking:

```bash
python tools/extract_avion_epic.py input.mp4 weights/epic_cache/input.npz
```

The extractor uses the local MD5-verified AVION ViT-B checkpoint and supports CPU, CUDA, and MPS. An NPZ cache contains 2-second and 4-second window timestamps plus verb/noun logits.

See [weights/README.md](weights/README.md) for checkpoint paths and cache details.

## Output format

Each prediction is written as JSON:

```json
{
  "team_id": "YOUR_TEAM_ID",
  "video_id": "P02_13",
  "duration_sec": 29.51,
  "label": {
    "name": "cooking",
    "score": 1.0
  },
  "actions": [],
  "highlights": []
}
```

The output assembler clamps segments to video duration, canonicalizes labels, and removes invalid or empty content. A representative validated result is available at [outputs/P02_13.json](outputs/P02_13.json).

## Validation

```bash
python -m unittest discover -s tests -v
python validate.py --dir outputs/
python tools/environment_check.py
```

- Unit tests cover fallback behavior, selection logic, cache integration, and synthetic videos.
- `validate.py` checks local submission shape and manifest coverage.
- `environment_check.py` reports package, hardware, and local model availability.
- It does not replace the official organizer validator or establish measured accuracy.

## Windows setup

The Windows setup script prepares CUDA packages, model dependencies, AVION, and local weights:

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_windows.ps1 `
  -WorkDir D:\Kairos-Track4-work `
  -TeamId YOUR_TEAM_ID
```

It uses the configured work directory for weights and large downloads while keeping Python packages in `.venv`.

## Current status

### Integrated

- 187-video manifest batch workflow
- Motion/audio/semantic fallback pipeline
- Official action vocabulary and EPIC cache contract
- AVION ViT-B checkpoint loading and synthetic forward pass
- FPVSum route, temporal segmentation, and knapsack highlight selection
- Offline model loading and disk-conscious source streaming
- Output assembly, validation, and representative output JSON

### Still unverified

- Real EPIC and FPVSum dataset accuracy
- Real-weight VideoMAE and SigLIP inference quality
- EPIC calibration and action coverage
- AVION throughput and benchmark mAP
- End-to-end performance under the official organizer validator

Measured accuracy should not be claimed until benchmark results are available.

## Documentation

- [Project status](docs/PROJECT_STATUS.md)
- [Model and weight layout](weights/README.md)
- [Track 4 manifest](manifest.csv)

## License

This repository does not currently include a license file. Add one before distributing the project or publishing derived datasets and model artifacts.

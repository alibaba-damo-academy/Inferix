# Video Drift Evaluation (`vde.py`)

## LV-Bench Dataset Overview
LV-Bench is a curated benchmark of 1,000 minute-long videos targeted at evaluating long-horizon generation. Videos are sourced from DanceTrack, GOT-10k, HD-VILA-100M, and ShareGPT4V, yielding a class distribution of roughly 67% human-focused, 17% animal-focused, and 16% environment-focused footage. Each source video is broken into 2–3 second segments and captioned with GPT-4o, followed by human validation at every stage (sourcing, chunking, caption review) to maintain quality. The benchmark is divided into an 80/20 train-eval split and pairs the VDE suite with standard VBench scores, providing a comprehensive stress test for temporal coherence.

This directory contains a single entry point, `vde.py`, that computes Video Drift Error (VDE) scores for every `.mp4` file inside a target directory. VDE provides a simple way to monitor how quality-related metrics drift across chunks of the same video. The script already supports several metric backends (clarity, motion, aesthetic, dynamic, subject, background) via the `vbench` tooling.

## LV-Bench Dataset
```
TODO
```

## Environment Setup
- Install the project dependencies inside your Conda environment (PyTorch, torchvision, OpenCV, NumPy, `vbench`, and the local `metrics` module must be importable).

- Install requirements from `requirements.txt`.

- Download the pre-trained weights according to the guidance in the `download.sh` file for each model (see each folder).

## Running the Evaluator
**Execute the script from the directory `LV-Bench`.** The command below processes every `.mp4` in the chosen input folder and writes one JSON file per metric into the output directory.

```bash
python vde.py --video_dir <your input folder> --output_dir <your output folder>
```

Each JSON file follows the pattern `vde_<metric_name>.json` and stores the per-video VDE scores. If a video contains fewer frames than the configured chunk count (defaults to 10), it is skipped with a warning.

## Configuration Notes
- To adjust which metrics run, update the `SUPPORTED_METRICS` list in `vde.py`.
- Modify `N_CHUNKS` to change the temporal resolution of chunking.
- Additional arguments required by specific metrics (for example, clarity’s `num_frames_to_sample`) can be supplied through the `kwargs` section in the main loop.

## Troubleshooting
- Ensure CUDA is available when running GPU-heavy metrics; the script falls back to CPU if CUDA is unavailable.
- The evaluator expects all metric factories to be registered in `metrics.create_metric_func`. Missing entries there will raise `NotImplementedError`.

## Video Drift Error Metrics
BlockVid introduces Video Drift Error (VDE) as a family of drift-aware metrics derived from Weighted Mean Absolute Percentage Error (WMAPE). Long videos are split into uniform temporal chunks; each chunk is scored with an underlying VBench metric, and VDE captures how far subsequent chunks deviate from the first chunk’s baseline score. Lower VDE values indicate better temporal stability.

- `VDE Clarity` tracks gradual loss of sharpness or resolution.
- `VDE Motion` watches for jitter, freezing, or other dynamics drift.
- `VDE Aesthetic` measures shifts in visual appeal or artistic coherence.
- `VDE Background` highlights scene or setting drift over time.
- `VDE Subject` monitors identity consistency of the main subject.

These metrics are aggregated with linear weights by default (see `vde()` in `vde.py`), but you can experiment with logarithmic weighting to emphasize late-chunk stability.

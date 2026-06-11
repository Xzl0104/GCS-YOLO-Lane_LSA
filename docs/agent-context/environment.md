# Environment

This file records the local Python/CUDA runtime expected by this project.

## Local CUDA Conda Environment

Project CUDA environment path:

```text
D:\miniconda3\envs\lsa_yolo
```

Environment name:

```text
lsa_yolo
```

Preferred Windows activation from a terminal:

```bat
conda activate lsa_yolo
```

Direct Python executable when activation is unavailable:

```bat
D:\miniconda3\envs\lsa_yolo\python.exe
```

Quick CUDA sanity check:

```bat
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

## Runtime Rules

- Run training, validation, inference, and contract checks from the repository root after activating `lsa_yolo`.
- Do not create or switch to a new Python environment unless the user explicitly asks.
- Do not hardcode this absolute Windows path into portable source code, dataset YAML, or committed scripts unless the user explicitly requests a local-only script.
- If a command fails because of CUDA, PyTorch, or environment mismatch, report the exact Python executable, PyTorch version, CUDA availability, and CUDA version before proposing fixes.
- Preserve the TuSimple input-size contract: `--imgsz 544 960`.

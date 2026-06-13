# Environment

This file records the local and remote Python/CUDA runtimes expected by this project.

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

## Remote CUDA Server Notes

When using a remote CUDA server, prefer a dedicated Git clone for the published source instead of overwriting a pre-existing non-Git training directory that may contain local datasets, runs, or checkpoints.

Keep SSH hosts, usernames, ports, passwords, keys, and exact session-only server paths out of committed documentation. Treat them as operator-provided session parameters.

The current remote CUDA server environment is:

```text
ssh_lane
```

Preferred activation on the remote server:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ssh_lane
```

Direct Python executable when activation is unavailable:

```text
/root/miniconda3/envs/ssh_lane/bin/python
```

Remote setup checks:

```bash
git pull --ff-only
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA')"
```

On the primary Windows workstation used for this project, the preferred remote login is the local SSH config alias:

```bash
ssh gcs-ebcloud-lane
```

This alias is configured outside the repository in the user's SSH config and should authenticate by SSH key. If the alias stops working, repair the local SSH config or server `authorized_keys`; do not add plaintext passwords to repository docs.

If the operator changes the environment name or shorthand path, verify the real Python executable on the server before running training or evaluation. Do not assume that `/miniconda3/...` and `$HOME/miniconda3/...` both exist.

Remote datasets and large local weights should be linked or copied into the dedicated Git clone as local runtime artifacts. Do not commit generated dataset archives, checkpoints, run logs, or transferred data packages.

## Runtime Rules

- For local Codex validation, inference smoke checks, and contract checks, run from the repository root after activating `lsa_yolo`.
- For server training, official-val evaluation, and longer CUDA experiments, connect with `ssh gcs-ebcloud-lane`, activate `ssh_lane`, and run from the dedicated remote Git clone.
- Do not create or switch to a new Python environment unless the user explicitly asks.
- Do not hardcode this absolute Windows path into portable source code, dataset YAML, or committed scripts unless the user explicitly requests a local-only script.
- If a command fails because of CUDA, PyTorch, or environment mismatch, report the exact Python executable, PyTorch version, CUDA availability, and CUDA version before proposing fixes.
- Preserve the TuSimple input-size contract: `--imgsz 544 960`.

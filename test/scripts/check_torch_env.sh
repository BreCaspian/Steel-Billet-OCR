#!/usr/bin/env bash
set -euo pipefail

echo "[1/2] Quick check"
python3 -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('cuda_count:', torch.cuda.device_count())"

echo
echo "[2/2] Full check"
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)
print("cudnn_version:", torch.backends.cudnn.version())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY

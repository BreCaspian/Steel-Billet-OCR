FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_VISIBLE_DEVICES="" \
    STAGE1_MODEL=/app/models/stage-1/Stage-1-S-base.pt \
    STAGE2_MODEL=/app/models/stage-2/Stage-2-S-base.pt \
    DATA_YAML=/app/configs/data-char.yaml \
    DEVICE=cpu \
    CONF1=0.25 \
    CONF2=0.50 \
    IOU=0.7 \
    EXPAND1=1.08 \
    PAD=0.10

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r /app/requirements.txt

COPY src /app/src
COPY test/scripts /app/test/scripts
COPY configs /app/configs
COPY models /app/models
COPY README.md /app/README.md

RUN test -f /app/configs/data-char.yaml

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

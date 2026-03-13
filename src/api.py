import base64
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest

from src.two_stage_engine import TwoStageOBBEngine, load_names_from_yaml


class OCRRequest(BaseModel):
    type: str
    images: str


app = FastAPI(title="Steel Billet OCR API (Two-Stage OBB)", version="2.0")

STAGE1_MODEL = os.environ.get("STAGE1_MODEL", "/app/models/stage-1/Stage-1-S-base.pt")
STAGE2_MODEL = os.environ.get("STAGE2_MODEL", "/app/models/stage-2/Stage-2-S-base.pt")
DATA_YAML = os.environ.get("DATA_YAML", "/app/configs/data-char.yaml")
DEVICE = os.environ.get("DEVICE", "cpu")
CONF1 = float(os.environ.get("CONF1", "0.25"))
CONF2 = float(os.environ.get("CONF2", "0.50"))
IOU = float(os.environ.get("IOU", "0.7"))
EXPAND1 = float(os.environ.get("EXPAND1", "1.08"))
PAD = float(os.environ.get("PAD", "0.10"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

_engine = None
_stage2_names = []

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ocr-api")

REQ_COUNT = Counter("ocr_requests_total", "Total OCR requests", ["status"])
REQ_LATENCY = Histogram("ocr_request_latency_seconds", "OCR request latency (seconds)")
DETECT_COUNT = Counter("ocr_detections_total", "Total detections", ["result"])


@app.on_event("startup")
def _startup():
    global _engine, _stage2_names
    _engine = TwoStageOBBEngine(
        stage1_model=STAGE1_MODEL,
        stage2_model=STAGE2_MODEL,
        device=DEVICE,
        conf1=CONF1,
        conf2=CONF2,
        iou=IOU,
        expand1=EXPAND1,
        pad=PAD,
    )
    _stage2_names = load_names_from_yaml(Path(DATA_YAML))
    logger.info(
        "Models loaded. stage1=%s stage2=%s data=%s device=%s conf1=%s conf2=%s",
        STAGE1_MODEL,
        STAGE2_MODEL,
        DATA_YAML,
        DEVICE,
        CONF1,
        CONF2,
    )


def _decode_base64_image(data: str):
    if not data:
        raise ValueError("empty image data")
    if "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data, validate=True)
    except Exception:
        raw = base64.b64decode(data + "===")

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode image")
    return img


@app.post("/ocr")
def ocr(req: OCRRequest, request: Request):
    if req.type.lower() != "base64":
        raise HTTPException(status_code=400, detail="type must be 'base64'")

    t0 = time.time()
    try:
        img = _decode_base64_image(req.images)
    except Exception as e:
        REQ_COUNT.labels(status="decode_error").inc()
        logger.warning("decode_error ip=%s err=%s", request.client.host if request.client else "unknown", e)
        return {"success": False, "errorMsg": f"decode_error: {e}"}

    result = _engine.infer_bgr(img)
    REQ_LATENCY.observe(time.time() - t0)

    if result.status != "ok":
        REQ_COUNT.labels(status=result.status).inc()
        logger.info(
            "fail ip=%s status=%s s1=%.4f s2=%.4f",
            request.client.host if request.client else "unknown",
            result.status,
            result.stage1_conf,
            result.stage2_avg_conf,
        )
        return {"success": False, "errorMsg": result.status}

    DETECT_COUNT.labels(result="ok").inc()
    REQ_COUNT.labels(status="ok").inc()
    logger.info(
        "ok ip=%s result=%s score=%.4f s1=%.4f s2=%.4f len=%d",
        request.client.host if request.client else "unknown",
        result.text,
        result.score,
        result.stage1_conf,
        result.stage2_avg_conf,
        len(result.char_labels),
    )

    return {
        "success": True,
        "result": result.text,
        "score": result.score,
    }


@app.get("/health")
def health():
    return {
        "success": True,
        "stage1_model": STAGE1_MODEL,
        "stage2_model": STAGE2_MODEL,
        "device": DEVICE,
        "stage2_classes": len(_stage2_names),
    }


@app.get("/metrics")
def metrics():
    return generate_latest()

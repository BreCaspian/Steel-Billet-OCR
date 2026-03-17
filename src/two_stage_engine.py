from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

Point = Tuple[float, float]
Quad = List[Point]
MappedChar = Tuple[Quad, int, float]


@dataclass
class TwoStageResult:
    text: str
    status: str
    stage1_conf: float
    stage2_avg_conf: float
    score: float
    stage1_quad: List[Point]
    char_quads: List[List[Point]]
    char_labels: List[str]
    char_scores: List[float]


class TwoStageOBBEngine:
    def __init__(
        self,
        stage1_model: str,
        stage2_model: str,
        device: str = "cpu",
        conf1: float = 0.25,
        conf2: float = 0.50,
        iou: float = 0.7,
        expand1: float = 1.08,
        pad: float = 0.10,
    ):
        self.model1 = YOLO(stage1_model, task="obb")
        self.model2 = YOLO(stage2_model, task="obb")
        self.device = device
        self.conf1 = conf1
        self.conf2 = conf2
        self.iou = iou
        self.expand1 = expand1
        self.pad = pad

    @staticmethod
    def _empty_result(status: str, stage1_conf: float = 0.0, stage1_quad: Sequence[Point] | None = None) -> TwoStageResult:
        return TwoStageResult(
            text="",
            status=status,
            stage1_conf=stage1_conf,
            stage2_avg_conf=0.0,
            score=0.0,
            stage1_quad=list(stage1_quad or []),
            char_quads=[],
            char_labels=[],
            char_scores=[],
        )

    @staticmethod
    def _to_quads(obb_obj: Any) -> List[Quad]:
        if obb_obj is None or len(obb_obj) == 0:
            return []
        quads = obb_obj.xyxyxyxy.cpu().tolist()
        return [[(float(px), float(py)) for px, py in quad] for quad in quads]

    @staticmethod
    def _order_quad(quad: Sequence[Point]) -> Quad:
        pts = np.array(quad, dtype=np.float32)
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        pts = pts[np.argsort(angles)]
        start = int(np.argmin(pts[:, 0] + pts[:, 1]))
        pts = np.vstack([pts[start:], pts[:start]])
        return [(float(x), float(y)) for x, y in pts]

    @staticmethod
    def _centroid(quad: Sequence[Point]) -> Point:
        return (sum(x for x, _ in quad) / 4.0, sum(y for _, y in quad) / 4.0)

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _expand_quad(self, quad: Sequence[Point], width: int, height: int) -> Quad:
        if self.expand1 <= 1.0:
            return [(float(x), float(y)) for x, y in quad]

        cx, cy = self._centroid(quad)
        expanded: Quad = []
        for x, y in quad:
            nx = cx + (x - cx) * self.expand1
            ny = cy + (y - cy) * self.expand1
            expanded.append((self._clamp(nx, 0, width - 1), self._clamp(ny, 0, height - 1)))
        return expanded

    def _build_warp(self, image_bgr: np.ndarray, quad: Sequence[Point]) -> Tuple[np.ndarray, np.ndarray]:
        ordered = np.array(self._order_quad(quad), dtype=np.float32)
        edge_w = max(np.linalg.norm(ordered[1] - ordered[0]), np.linalg.norm(ordered[2] - ordered[3]), 2.0)
        edge_h = max(np.linalg.norm(ordered[2] - ordered[1]), np.linalg.norm(ordered[3] - ordered[0]), 2.0)
        pad_x = edge_w * self.pad
        pad_y = edge_h * self.pad
        out_w = int(max(8, round(edge_w + 2 * pad_x)))
        out_h = int(max(8, round(edge_h + 2 * pad_y)))

        dst = np.array(
            [
                [pad_x, pad_y],
                [pad_x + edge_w, pad_y],
                [pad_x + edge_w, pad_y + edge_h],
                [pad_x, pad_y + edge_h],
            ],
            dtype=np.float32,
        )

        h_mat = cv2.getPerspectiveTransform(ordered, dst)
        h_inv = cv2.getPerspectiveTransform(dst, ordered)
        warped = cv2.warpPerspective(
            image_bgr,
            h_mat,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return warped, h_inv

    @staticmethod
    def _map_quad(quad: Sequence[Point], h_inv: np.ndarray) -> Quad:
        pts = np.array(quad, dtype=np.float32).reshape(1, 4, 2)
        mapped = cv2.perspectiveTransform(pts, h_inv).reshape(4, 2)
        return [(float(x), float(y)) for x, y in mapped]

    @staticmethod
    def _predict_obb(model: YOLO, image_bgr: np.ndarray, conf: float, iou: float, device: str):
        return model.predict(
            source=image_bgr,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )[0]

    def _sort_chars(self, quads_with_cls: Sequence[MappedChar]) -> List[MappedChar]:
        return sorted(quads_with_cls, key=lambda item: (self._centroid(item[0])[0], self._centroid(item[0])[1]))

    @staticmethod
    def _stage2_names(result: Any) -> dict[int, str]:
        return {int(k): str(v) for k, v in getattr(result, "names", {}).items()}

    def infer_bgr(self, image_bgr: np.ndarray) -> TwoStageResult:
        if image_bgr is None or image_bgr.size == 0:
            return self._empty_result("read_failed")

        height, width = image_bgr.shape[:2]
        stage1_result = self._predict_obb(self.model1, image_bgr, self.conf1, self.iou, self.device)
        stage1_obb = stage1_result.obb
        if stage1_obb is None or len(stage1_obb) == 0:
            return self._empty_result("stage1_no_detection")

        stage1_scores = stage1_obb.conf.cpu().tolist()
        stage1_quads = self._to_quads(stage1_obb)
        best_idx = max(range(len(stage1_scores)), key=stage1_scores.__getitem__)
        stage1_conf = float(stage1_scores[best_idx])
        stage1_quad = self._expand_quad(stage1_quads[best_idx], width, height)

        try:
            warp_bgr, h_inv = self._build_warp(image_bgr, stage1_quad)
        except Exception:
            return self._empty_result("warp_failed", stage1_conf=stage1_conf, stage1_quad=stage1_quad)

        stage2_result = self._predict_obb(self.model2, warp_bgr, self.conf2, self.iou, self.device)
        stage2_obb = stage2_result.obb
        if stage2_obb is None or len(stage2_obb) == 0:
            return self._empty_result("stage2_no_detection", stage1_conf=stage1_conf, stage1_quad=stage1_quad)

        names2 = self._stage2_names(stage2_result)
        quads2 = self._to_quads(stage2_obb)
        cls2 = [int(v) for v in stage2_obb.cls.cpu().tolist()]
        conf2 = [float(v) for v in stage2_obb.conf.cpu().tolist()]

        mapped_chars = [
            (self._map_quad(quad, h_inv), cls_id, score)
            for quad, cls_id, score in zip(quads2, cls2, conf2)
        ]
        ordered_chars = self._sort_chars(mapped_chars)

        labels = [names2.get(cls_id, str(cls_id)) for _, cls_id, _ in ordered_chars]
        scores = [score for _, _, score in ordered_chars]
        text = "".join(labels)
        stage2_avg_conf = float(sum(scores) / len(scores)) if scores else 0.0
        final_score = min(stage1_conf, stage2_avg_conf) if scores else 0.0

        return TwoStageResult(
            text=text,
            status="ok" if text else "stage2_no_detection",
            stage1_conf=stage1_conf,
            stage2_avg_conf=stage2_avg_conf,
            score=final_score,
            stage1_quad=stage1_quad,
            char_quads=[quad for quad, _, _ in ordered_chars],
            char_labels=labels,
            char_scores=scores,
        )


def load_names_from_yaml(data_yaml: Path) -> List[str]:
    if not data_yaml.exists():
        return []

    lines = data_yaml.read_text(encoding="utf-8").splitlines()
    names: List[str] = []
    indexed: dict[int, str] = {}
    in_names = False

    for line in lines:
        raw = line.strip()
        if raw.startswith("names:"):
            in_names = True
            continue
        if not in_names or not raw:
            continue
        if raw.startswith("-"):
            names.append(raw[1:].strip())
            continue
        if ":" in raw and raw.split(":", 1)[0].isdigit():
            k, v = raw.split(":", 1)
            indexed[int(k.strip())] = v.strip().strip('"').strip("'")
            continue
        if not line.startswith(" "):
            break

    if names:
        return names
    if indexed:
        return [indexed[i] for i in sorted(indexed)]
    return []


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

Point = Tuple[float, float]


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
    def _to_quads(obb_obj) -> List[List[Point]]:
        if obb_obj is None or len(obb_obj) == 0:
            return []
        quads = obb_obj.xyxyxyxy.cpu().tolist()
        return [[(float(p[0]), float(p[1])) for p in q] for q in quads]

    @staticmethod
    def _order_quad(quad: Sequence[Point]) -> List[Point]:
        pts = np.array(quad, dtype=np.float32)
        c = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        pts = pts[np.argsort(angles)]
        start = int(np.argmin(pts[:, 0] + pts[:, 1]))
        pts = np.vstack([pts[start:], pts[:start]])
        return [(float(x), float(y)) for x, y in pts]

    @staticmethod
    def _centroid(quad: Sequence[Point]) -> Point:
        return (sum(p[0] for p in quad) / 4.0, sum(p[1] for p in quad) / 4.0)

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def _expand_quad(self, quad: Sequence[Point], w: int, h: int) -> List[Point]:
        if self.expand1 <= 1.0:
            return [(float(x), float(y)) for x, y in quad]
        cx = sum(p[0] for p in quad) / 4.0
        cy = sum(p[1] for p in quad) / 4.0
        out = []
        for x, y in quad:
            nx = cx + (x - cx) * self.expand1
            ny = cy + (y - cy) * self.expand1
            out.append((self._clamp(nx, 0, w - 1), self._clamp(ny, 0, h - 1)))
        return out

    def _build_warp(self, image_bgr: np.ndarray, quad: Sequence[Point]):
        q = np.array(self._order_quad(quad), dtype=np.float32)
        edge_w = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]), 2.0)
        edge_h = max(np.linalg.norm(q[2] - q[1]), np.linalg.norm(q[3] - q[0]), 2.0)
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

        h_mat = cv2.getPerspectiveTransform(q, dst)
        h_inv = cv2.getPerspectiveTransform(dst, q)
        warped = cv2.warpPerspective(
            image_bgr,
            h_mat,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return warped, h_inv

    @staticmethod
    def _map_quad(quad: Sequence[Point], h_inv: np.ndarray) -> List[Point]:
        pts = np.array(quad, dtype=np.float32).reshape(1, 4, 2)
        mapped = cv2.perspectiveTransform(pts, h_inv).reshape(4, 2)
        return [(float(x), float(y)) for x, y in mapped]

    def _sort_chars(self, quads_with_cls, string_quad: Sequence[Point]):
        ax = string_quad[1][0] - string_quad[0][0]
        ay = string_quad[1][1] - string_quad[0][1]
        n = (ax * ax + ay * ay) ** 0.5
        axis = (1.0, 0.0) if n < 1e-6 else (ax / n, ay / n)
        origin = self._centroid(string_quad)
        return sorted(
            quads_with_cls,
            key=lambda t: (self._centroid(t[0])[0] - origin[0]) * axis[0]
            + (self._centroid(t[0])[1] - origin[1]) * axis[1],
        )

    def infer_bgr(self, image_bgr: np.ndarray) -> TwoStageResult:
        if image_bgr is None or image_bgr.size == 0:
            return TwoStageResult("", "read_failed", 0.0, 0.0, 0.0, [], [], [], [])

        h, w = image_bgr.shape[:2]

        r1 = self.model1.predict(
            source=image_bgr,
            conf=self.conf1,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]

        obb1 = r1.obb
        if obb1 is None or len(obb1) == 0:
            return TwoStageResult("", "stage1_no_detection", 0.0, 0.0, 0.0, [], [], [], [])

        confs1 = obb1.conf.cpu().tolist()
        quads1 = self._to_quads(obb1)
        best_idx = max(range(len(confs1)), key=lambda i: confs1[i])
        stage1_quad = self._expand_quad(quads1[best_idx], w, h)
        stage1_conf = float(confs1[best_idx])

        try:
            warp_bgr, h_inv = self._build_warp(image_bgr, stage1_quad)
        except Exception:
            return TwoStageResult("", "warp_failed", stage1_conf, 0.0, 0.0, stage1_quad, [], [], [])

        r2 = self.model2.predict(
            source=warp_bgr,
            conf=self.conf2,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]

        obb2 = r2.obb
        if obb2 is None or len(obb2) == 0:
            return TwoStageResult("", "stage2_no_detection", stage1_conf, 0.0, 0.0, stage1_quad, [], [], [])

        names2 = {int(k): str(v) for k, v in getattr(r2, "names", {}).items()}
        quads2 = self._to_quads(obb2)
        cls2 = [int(v) for v in obb2.cls.cpu().tolist()]
        conf2 = [float(v) for v in obb2.conf.cpu().tolist()]

        mapped = [(self._map_quad(q, h_inv), c, s) for q, c, s in zip(quads2, cls2, conf2)]
        ordered = self._sort_chars(mapped, stage1_quad)

        labels = [names2.get(c, str(c)) for _, c, _ in ordered]
        scores = [s for _, _, s in ordered]
        text = "".join(labels)
        stage2_avg = float(sum(scores) / len(scores)) if scores else 0.0
        final_score = min(stage1_conf, stage2_avg) if scores else 0.0
        return TwoStageResult(
            text=text,
            status="ok" if text else "stage2_no_detection",
            stage1_conf=stage1_conf,
            stage2_avg_conf=stage2_avg,
            score=final_score,
            stage1_quad=stage1_quad,
            char_quads=[q for q, _, _ in ordered],
            char_labels=labels,
            char_scores=scores,
        )


def load_names_from_yaml(data_yaml: Path) -> List[str]:
    if not data_yaml.exists():
        return []

    lines = data_yaml.read_text(encoding="utf-8").splitlines()
    names: List[str] = []
    in_names = False
    indexed = {}

    for line in lines:
        raw = line.strip()
        if raw.startswith("names:"):
            in_names = True
            continue
        if not in_names:
            continue
        if not raw:
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
        return [indexed[i] for i in sorted(indexed.keys())]
    return []

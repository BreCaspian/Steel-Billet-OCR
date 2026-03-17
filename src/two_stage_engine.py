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
Box = Tuple[float, float, float, float]


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
        conf2: float = 0.6,
        iou: float = 0.40,
        expand1: float = 1.50,
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

    @staticmethod
    def _predict_obb(model: YOLO, image_bgr: np.ndarray, conf: float, iou: float, device: str):
        return model.predict(
            source=image_bgr,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )[0]

    @staticmethod
    def _point_in_quad(point: Point, quad: Sequence[Point]) -> bool:
        contour = np.array(quad, dtype=np.float32).reshape((-1, 1, 2))
        return cv2.pointPolygonTest(contour, point, False) >= 0

    def _filter_chars_in_stage1(self, chars: Sequence[MappedChar], stage1_quad: Sequence[Point]) -> List[MappedChar]:
        kept: List[MappedChar] = []
        for char_quad, cls_id, score in chars:
            if self._point_in_quad(self._centroid(char_quad), stage1_quad):
                kept.append((char_quad, cls_id, score))
        return kept

    @staticmethod
    def _quad_to_box(quad: Sequence[Point]) -> Box:
        xs = [x for x, _ in quad]
        ys = [y for _, y in quad]
        return (min(xs), min(ys), max(xs), max(ys))

    @staticmethod
    def _chars_to_box(chars: Sequence[MappedChar]) -> Box | None:
        if not chars:
            return None
        xs: List[float] = []
        ys: List[float] = []
        for quad, _, _ in chars:
            for x, y in quad:
                xs.append(x)
                ys.append(y)
        return (min(xs), min(ys), max(xs), max(ys))

    @staticmethod
    def _box_to_quad(box: Box) -> Quad:
        x1, y1, x2, y2 = box
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    @staticmethod
    def _box_area(box: Box) -> float:
        x1, y1, x2, y2 = box
        return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)

    @staticmethod
    def _intersection_area(box1: Box, box2: Box) -> float:
        ax1, ay1, ax2, ay2 = box1
        bx1, by1, bx2, by2 = box2
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        return max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)

    @staticmethod
    def _char_size(char: MappedChar) -> Tuple[float, float]:
        x1, y1, x2, y2 = TwoStageOBBEngine._quad_to_box(char[0])
        return (max(x2 - x1, 1.0), max(y2 - y1, 1.0))

    def _select_stage2_cluster(self, chars: Sequence[MappedChar], stage1_quad: Sequence[Point]) -> List[MappedChar]:
        if not chars:
            return []

        stage1_cx, stage1_cy = self._centroid(stage1_quad)
        stage1_x1, stage1_y1, stage1_x2, stage1_y2 = self._quad_to_box(stage1_quad)
        stage1_w = max(stage1_x2 - stage1_x1, 1.0)
        stage1_h = max(stage1_y2 - stage1_y1, 1.0)

        scored: List[Tuple[float, MappedChar]] = []
        for char in chars:
            cx, cy = self._centroid(char[0])
            cw, ch = self._char_size(char)
            vertical_penalty = abs(cy - stage1_cy) / max(stage1_h, ch)
            horizontal_penalty = abs(cx - stage1_cx) / stage1_w
            score = char[2] - 0.35 * vertical_penalty - 0.10 * horizontal_penalty
            scored.append((score, char))

        seed = max(scored, key=lambda item: item[0])[1]
        seed_cx, seed_cy = self._centroid(seed[0])
        seed_w, seed_h = self._char_size(seed)
        max_center_gap_x = max(seed_w * 2.5, stage1_w * 0.18)
        max_center_gap_y = max(seed_h * 1.2, stage1_h * 0.20)

        cluster: List[MappedChar] = []
        for char in chars:
            cx, cy = self._centroid(char[0])
            if abs(cy - seed_cy) <= max_center_gap_y and abs(cx - seed_cx) <= stage1_w * 0.75:
                cluster.append(char)

        cluster.sort(key=lambda item: self._centroid(item[0])[0])
        if not cluster:
            return [seed]

        refined = [cluster[0]]
        for char in cluster[1:]:
            prev_cx, prev_cy = self._centroid(refined[-1][0])
            cx, cy = self._centroid(char[0])
            if abs(cy - prev_cy) <= max_center_gap_y and (cx - prev_cx) <= max_center_gap_x:
                refined.append(char)
        return refined if refined else [seed]

    def _select_final_stage1_quad(
        self,
        stage1_raw_quad: Sequence[Point],
        chars: Sequence[MappedChar],
        width: int,
        height: int,
    ) -> Quad:
        stage1_box = self._quad_to_box(stage1_raw_quad)
        chars_box = self._chars_to_box(chars)
        if chars_box is None:
            return self._expand_quad(stage1_raw_quad, width, height)

        overlap = self._intersection_area(stage1_box, chars_box)
        stage1_area = max(self._box_area(stage1_box), 1.0)
        overlap_ratio = overlap / stage1_area

        if overlap_ratio > 0.80:
            base_quad = list(stage1_raw_quad)
        else:
            chars_quad = self._box_to_quad(chars_box)
            base_quad = list(stage1_raw_quad) if self._box_area(stage1_box) >= self._box_area(chars_box) else chars_quad

        return self._expand_quad(base_quad, width, height)

    def _sort_chars(self, quads_with_cls: Sequence[MappedChar]) -> List[MappedChar]:
        return sorted(quads_with_cls, key=lambda item: self._centroid(item[0])[0])

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
        stage1_raw_quad = stage1_quads[best_idx]

        stage2_result = self._predict_obb(self.model2, image_bgr, self.conf2, self.iou, self.device)
        stage2_obb = stage2_result.obb
        if stage2_obb is None or len(stage2_obb) == 0:
            return self._empty_result(
                "stage2_no_detection",
                stage1_conf=stage1_conf,
                stage1_quad=self._expand_quad(stage1_raw_quad, width, height),
            )

        names2 = self._stage2_names(stage2_result)
        quads2 = self._to_quads(stage2_obb)
        cls2 = [int(v) for v in stage2_obb.cls.cpu().tolist()]
        scores2 = [float(v) for v in stage2_obb.conf.cpu().tolist()]

        all_chars = [(quad, cls_id, score) for quad, cls_id, score in zip(quads2, cls2, scores2)]
        stage2_cluster = self._select_stage2_cluster(all_chars, stage1_raw_quad)
        if not stage2_cluster:
            return self._empty_result(
                "stage2_no_detection",
                stage1_conf=stage1_conf,
                stage1_quad=self._expand_quad(stage1_raw_quad, width, height),
            )

        final_stage1_quad = self._select_final_stage1_quad(stage1_raw_quad, stage2_cluster, width, height)
        filtered_chars = self._filter_chars_in_stage1(all_chars, final_stage1_quad)
        if not filtered_chars:
            return self._empty_result("stage2_no_detection", stage1_conf=stage1_conf, stage1_quad=final_stage1_quad)

        ordered_chars = self._sort_chars(filtered_chars)
        labels = [names2.get(cls_id, str(cls_id)) for _, cls_id, _ in ordered_chars]
        ordered_scores = [score for _, _, score in ordered_chars]
        text = "".join(labels)
        stage2_avg_conf = float(sum(ordered_scores) / len(ordered_scores)) if ordered_scores else 0.0
        final_score = min(stage1_conf, stage2_avg_conf) if ordered_scores else 0.0

        return TwoStageResult(
            text=text,
            status="ok" if text else "stage2_no_detection",
            stage1_conf=stage1_conf,
            stage2_avg_conf=stage2_avg_conf,
            score=final_score,
            stage1_quad=final_stage1_quad,
            char_quads=[quad for quad, _, _ in ordered_chars],
            char_labels=labels,
            char_scores=ordered_scores,
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


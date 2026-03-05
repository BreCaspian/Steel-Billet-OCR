from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

DEFAULT_CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "B", "C", "D", "G", "H", "S", "W", "X"]


def convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def rotate_points(points: List[Tuple[float, float]], theta: float) -> List[Tuple[float, float]]:
    c = math.cos(theta)
    s = math.sin(theta)
    return [(x * c - y * s, x * s + y * c) for x, y in points]


def min_area_rect(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    hull = convex_hull(points)
    if len(hull) < 3:
        raise ValueError("polygon has fewer than 3 unique points")

    best = None
    best_area = float("inf")
    best_theta = 0.0

    for i in range(len(hull)):
        p1 = hull[i]
        p2 = hull[(i + 1) % len(hull)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        theta = math.atan2(edge[1], edge[0])
        rot = rotate_points(hull, -theta)
        xs = [p[0] for p in rot]
        ys = [p[1] for p in rot]
        min_x, min_y = min(xs), min(ys)
        max_x, max_y = max(xs), max(ys)
        area = (max_x - min_x) * (max_y - min_y)
        if area < best_area:
            best_area = area
            best_theta = theta
            best = (min_x, min_y, max_x, max_y)

    min_x, min_y, max_x, max_y = best
    corners_rot = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
    return rotate_points(corners_rot, best_theta)


def order_quad(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    pts = sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    start = min(range(len(pts)), key=lambda i: (pts[i][1], pts[i][0]))
    return pts[start:] + pts[:start]


def parse_points(raw_points: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    pts = []
    for p in raw_points:
        if not isinstance(p, list) or len(p) < 2:
            raise ValueError("point item must be [x, y]")
        x, y = p[0], p[1]
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError("point coordinate must be numeric")
        pts.append((float(x), float(y)))
    return pts


def convert_one_json(json_path: Path, out_txt_path: Path, class_to_id: Dict[str, int]) -> int:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    w = data["imageWidth"]
    h = data["imageHeight"]
    if not isinstance(w, (int, float)) or not isinstance(h, (int, float)) or w <= 0 or h <= 0:
        raise ValueError(f"{json_path.name}: invalid image size ({w}, {h})")

    lines: List[str] = []
    for idx, shape in enumerate(data["shapes"]):
        label = str(shape.get("label", "")).strip()
        if label == "00":
            label = "0"
        if label not in class_to_id:
            raise ValueError(f"{json_path.name}: unknown label '{label}' at shape #{idx}")

        pts = parse_points(shape.get("points", []))
        if len(pts) == 5:
            pts = min_area_rect(pts)
        elif len(pts) != 4:
            raise ValueError(f"{json_path.name}: shape #{idx} has {len(pts)} points, only 4/5 supported")

        pts = order_quad(pts)
        norm = []
        for x, y in pts:
            nx = min(max(x / float(w), 0.0), 1.0)
            ny = min(max(y / float(h), 0.0), 1.0)
            norm.extend([nx, ny])

        class_id = class_to_id[label]
        lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in norm))

    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    out_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Convert LabelMe JSON polygon labels to YOLO-OBB TXT")
    p.add_argument("--json-dir", required=True, help="Directory containing LabelMe JSON files")
    p.add_argument("--out-dir", required=True, help="Output directory for YOLO OBB TXT labels")
    p.add_argument("--classes", default=",".join(DEFAULT_CLASSES), help="Comma-separated class list")
    args = p.parse_args()

    json_dir = Path(args.json_dir)
    out_dir = Path(args.out_dir)
    classes = [x.strip() for x in args.classes.split(",") if x.strip()]
    class_to_id = {c: i for i, c in enumerate(classes)}

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"no json files found in {json_dir}")

    total_objects = 0
    for pth in json_files:
        total_objects += convert_one_json(pth, out_dir / f"{pth.stem}.txt", class_to_id)

    (out_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    (out_dir / "class_to_id.json").write_text(json.dumps(class_to_id, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"files": len(json_files), "objects": total_objects, "classes": classes}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

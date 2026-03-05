from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from src.two_stage_engine import TwoStageOBBEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stage OBB batch inference")
    p.add_argument("--stage1", default="models/Stage-1.pt", help="Stage-1 model (.pt)")
    p.add_argument("--stage2", default="models/Stage-2.pt", help="Stage-2 model (.pt)")
    p.add_argument("--source", default="images", help="Image directory")
    p.add_argument("--output", default="output", help="Output directory")
    p.add_argument("--device", default="cpu", help="cpu|0|1...")
    p.add_argument("--conf1", type=float, default=0.25)
    p.add_argument("--conf2", type=float, default=0.55)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--expand1", type=float, default=1.08)
    p.add_argument("--pad", type=float, default=0.10)
    p.add_argument("--save_vis", action="store_true", help="Save annotated images")
    return p.parse_args()


def draw_result(img_bgr, result):
    out = img_bgr.copy()
    if result.stage1_quad:
        pts = np.array([[int(x), int(y)] for x, y in result.stage1_quad], dtype=np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 0), 3)
    for quad, label, score in zip(result.char_quads, result.char_labels, result.char_scores):
        pts = np.array([[int(x), int(y)] for x, y in quad], dtype=np.int32)
        cv2.polylines(out, [pts], True, (0, 140, 255), 2)
        cx = int(sum(x for x, _ in quad) / 4)
        cy = int(sum(y for _, y in quad) / 4)
        cv2.putText(out, f"{label}:{score:.2f}", (cx, max(0, cy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(
        out,
        f"{result.status} text={result.text} score={result.score:.3f}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0) if result.status == "ok" else (0, 0, 255),
        2,
    )
    return out


def main() -> None:
    args = parse_args()

    src = Path(args.source)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    vis_dir = out / "annotated"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    engine = TwoStageOBBEngine(
        stage1_model=args.stage1,
        stage2_model=args.stage2,
        device=args.device,
        conf1=args.conf1,
        conf2=args.conf2,
        iou=args.iou,
        expand1=args.expand1,
        pad=args.pad,
    )

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in src.iterdir() if p.is_file() and p.suffix.lower() in exts])

    rows = []
    for i, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        result = engine.infer_bgr(img)
        rows.append(
            [
                img_path.name,
                result.text,
                result.status,
                f"{result.stage1_conf:.4f}",
                f"{result.stage2_avg_conf:.4f}",
                f"{result.score:.4f}",
            ]
        )
        print(f"[{i}/{len(images)}] {img_path.name} -> {result.status} text={result.text}")

        if args.save_vis and img is not None:
            ann = draw_result(img, result)
            cv2.imwrite(str(vis_dir / img_path.name), ann)

    out_csv = out / "predictions.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "recognized_text", "status", "stage1_conf", "stage2_avg_conf", "score"])
        w.writerows(rows)

    print(f"done. images={len(images)} csv={out_csv}")


if __name__ == "__main__":
    main()

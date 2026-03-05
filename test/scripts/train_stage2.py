from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train stage-2 OBB character detector")
    p.add_argument("--data", required=True, help="Stage-2 data.yaml")
    p.add_argument("--model", default="yolo11s-obb.pt", help="Model config/weights")
    p.add_argument("--init-weights", default="", help="Optional pretrained weights to load before train")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="0", help="GPU id or cpu")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="stage2_obb")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    model = YOLO(args.model, task="obb")
    if args.init_weights:
        init_path = Path(args.init_weights)
        if not init_path.exists():
            raise FileNotFoundError(f"init weights not found: {init_path}")
        model.load(str(init_path))

    kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "workers": args.workers,
        "patience": args.patience,
        "resume": args.resume,
        "seed": args.seed,
    }
    if args.amp:
        kwargs["amp"] = True

    model.train(**kwargs)


if __name__ == "__main__":
    main()

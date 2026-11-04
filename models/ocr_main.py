"""독립 OCR의 명시적 정답 출처·분할을 기록하고 CRNN 학습을 실행한다."""

import argparse
import json
from pathlib import Path

import pandas as pd

from models.ocr import build_imprint_dictionary, train_ocr
from src.config import PROJECT_ROOT, get_device
from src.ocr_data import read_ocr_review_table, reviewed_ocr_splits


def main() -> None:
    """검수표는 바꾸지 않고 별도 OCR 실행 폴더에 분할·설정·모델을 저장한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review", type=Path, default=PROJECT_ROOT / "data/processed/02_eda/ocr/ocr_review.csv"
    )
    parser.add_argument(
        "--selection", type=Path, default=PROJECT_ROOT / "data/processed/02_eda_selection.csv"
    )
    parser.add_argument("--split", default="combofix_v1_s42")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--external-val-fraction", type=float, default=0.0)
    parser.add_argument("--label-mode", choices=["reviewed", "metadata"], required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-cpu-training", action="store_true")
    args = parser.parse_args()
    review = read_ocr_review_table(args.review)
    selection = pd.read_csv(args.selection)
    split_path = PROJECT_ROOT / "data/processed/splits" / args.split / f"fold_{args.fold}.csv"
    rows = reviewed_ocr_splits(
        review,
        split_path,
        set(selection["file_name"]),
        label_mode=args.label_mode,
        external_val_fraction=args.external_val_fraction,
        seed=args.seed,
    )
    summary = {
        "label_mode": args.label_mode,
        "fold": args.fold,
        "split": args.split,
        "counts": rows.groupby("split").size().to_dict(),
        "label_sources": rows["label_source"].value_counts().to_dict(),
        "review_file_modified": False,
        "external_val_fraction": args.external_val_fraction,
        "image_counts": rows.groupby(["data_source", "split"])["file_name"].nunique().to_dict(),
    }
    summary["image_counts"] = {
        f"{source}/{split}": count for (source, split), count in summary["image_counts"].items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.loc[
        rows["data_source"].eq("external"), ["file_name", "combo", "split"]
    ].drop_duplicates().to_csv(args.output_dir / "external_split.csv", index=False)
    rows.to_csv(args.output_dir / "label_split.csv", index=False)
    (args.output_dir / "data_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    dictionary = build_imprint_dictionary(rows, label_mode=args.label_mode)
    (args.output_dir / "imprint_dictionary.json").write_text(
        json.dumps(dictionary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.prepare_only:
        return
    if get_device() == "cpu" and not args.allow_cpu_training:
        raise RuntimeError("현재 프로세스에서 GPU를 사용할 수 없습니다. GPU 커널에서 실행하세요.")
    checkpoint = train_ocr(
        rows,
        args.output_dir,
        epochs=args.epochs,
        height=args.height,
        width=args.width,
        batch=args.batch,
        lr=args.lr,
        seed=args.seed,
        workers=args.workers,
        hidden_size=args.hidden_size,
        label_mode=args.label_mode,
    )
    print(f"OCR 학습 완료: {checkpoint}", flush=True)


if __name__ == "__main__":
    main()

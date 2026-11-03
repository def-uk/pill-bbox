"""노트북의 YOLO 학습 상태와 무관하게 OCR 실험 입력과 결과를 준비한다."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from data_process.data_yaml import load_class_map, yolo_dir
from models.ocr import (
    build_imprint_dictionary,
    evaluate_ocr_predictions,
    predict_with_ocr,
    train_ocr,
)
from src.config import PROJECT_ROOT, get_device
from src.ocr_data import read_ocr_review_table, reviewed_ocr_splits


def prepare_ocr_context(
    yolo_weights: Path,
    review_path: Path,
    *,
    label: str,
    split: str,
    fold: int,
    external_val_fraction: float,
    seed: int,
    label_mode: str,
) -> dict:
    """기존 검수표·분할·라벨만 읽어 OCR 학습과 평가 입력을 준비한다."""
    split_path = PROJECT_ROOT / "data/processed/splits" / split / f"fold_{fold}.csv"
    for path in [yolo_weights, review_path, split_path]:
        if not path.is_file():
            raise FileNotFoundError(f"OCR 실행에 필요한 기존 파일이 없습니다: {path}")
    review = read_ocr_review_table(review_path)
    split_table = pd.read_csv(split_path)
    competition_files = set(split_table["file_name"])
    if "metadata_origin" not in review:
        raise ValueError("기존 OCR 검수표에 metadata_origin 열이 필요합니다.")
    external_files = set(review.loc[review["metadata_origin"].eq("aihub_json"), "file_name"])
    rows = reviewed_ocr_splits(
        review,
        split_path,
        external_files,
        label_mode=label_mode,
        external_val_fraction=external_val_fraction,
        seed=seed,
    )
    base = load_class_map(label)
    # 학습 시 저장한 외부 클래스 표에서 번호를 재구성하고 체크포인트와 대조한다.
    external_path = PROJECT_ROOT / "data/processed/yolo/v1_aihub_reviewed/aihub_labels.csv"
    model = YOLO(yolo_weights)
    class_map = base.copy()
    if len(model.names) > len(base):
        external = pd.read_csv(external_path)
        extra = external.loc[~external["category_id"].isin(base["category_id"])]
        extra = extra.drop_duplicates("category_id").sort_values("category_id")
        extra = extra[["category_id", "dl_name"]].copy()
        extra["yolo_idx"] = range(len(base), len(base) + len(extra))
        class_map = pd.concat([base, extra], ignore_index=True)
    expected = dict(zip(class_map["yolo_idx"], class_map["dl_name"], strict=True))
    if model.names != expected:
        raise ValueError("저장된 YOLO 클래스와 기존 클래스 표가 다릅니다.")
    del model
    truth = pd.read_csv(PROJECT_ROOT / "data/processed/labels" / f"labels_{label}.csv")
    val_files = set(split_table.loc[split_table["split"].eq("val"), "file_name"])
    if not val_files.issubset(set(truth["file_name"])):
        raise ValueError("기존 분할의 validation 정답이 누락됐습니다.")
    if external_files & competition_files:
        raise ValueError("대회와 외부 데이터 파일명이 겹칩니다.")
    return dict(
        rows=rows,
        class_map=class_map,
        base_class_map=base,
        val_truth=truth.loc[truth["file_name"].isin(val_files)].copy(),
        val_images=[yolo_dir(label) / "images/train" / name for name in sorted(val_files)],
        test_images=sorted(
            path
            for path in (yolo_dir(label) / "images/test").iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ),
    )


def run_ocr_training(
    context: dict,
    output_root: Path,
    *,
    fold: int,
    label_mode: str,
    **train_args,
) -> Path:
    """기존 모델을 보존하고 이번 분할·사전·OCR 학습 결과를 새 폴더에 저장한다."""
    if get_device() == "cpu":
        raise RuntimeError("GPU가 인식되지 않습니다. GPU를 사용할 수 있는 커널에서 실행하세요.")
    output_dir = output_root / f"fold{fold}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = context["rows"]
    rows.to_csv(output_dir / "label_split.csv", index=False)
    rows.loc[
        rows["data_source"].eq("external"), ["file_name", "combo", "split"]
    ].drop_duplicates().to_csv(
        output_dir / "external_split.csv",
        index=False,
    )
    dictionary = build_imprint_dictionary(rows, label_mode=label_mode)
    (output_dir / "imprint_dictionary.json").write_text(
        json.dumps(dictionary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return train_ocr(rows, output_dir, label_mode=label_mode, **train_args)


def run_ocr_prediction(
    context: dict,
    yolo_weights: Path,
    ocr_weights: Path,
    *,
    split: str,
    **predict_args,
) -> tuple[Path, pd.DataFrame | None]:
    """동일 검증 이미지의 비교 결과 또는 test 제출 CSV를 한 장씩 예측해 저장한다."""
    from models.evaluate import to_submission

    if split not in {"val", "test"}:
        raise ValueError("예측 대상은 val 또는 test를 지정하세요.")
    dictionary = {
        int(category): texts
        for category, texts in json.loads(
            (ocr_weights.parent / "imprint_dictionary.json").read_text(encoding="utf-8")
        ).items()
    }
    output_dir = ocr_weights.parent / f"predictions_{split}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    original, adjusted = predict_with_ocr(
        yolo_weights,
        ocr_weights,
        context[f"{split}_images"],
        context["class_map"],
        dictionary,
        output_dir,
        **predict_args,
    )
    (output_dir / "prediction_config.json").write_text(
        json.dumps(
            {
                "yolo_weights": str(yolo_weights.resolve()),
                "ocr_weights": str(ocr_weights.resolve()),
                "split": split,
                "image_count": len(context[f"{split}_images"]),
                **predict_args,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if split == "test":
        for name, predictions in [("yolo", original), ("ocr", adjusted)]:
            selected = predictions.loc[
                predictions["category_id"].isin(context["base_class_map"]["category_id"])
            ].copy()
            submission = to_submission(selected)
            if submission.empty or submission.isna().any().any():
                raise ValueError("제출 CSV가 비었거나 결측값이 있습니다.")
            submission.to_csv(output_dir / f"submission_{name}.csv", index=False)
        return output_dir, None
    metrics = pd.DataFrame(
        {
            "YOLO": evaluate_ocr_predictions(original, context["val_truth"]),
            "YOLO + OCR": evaluate_ocr_predictions(adjusted, context["val_truth"]),
        }
    ).T
    metrics.to_csv(output_dir / "validation_comparison.csv")
    return output_dir, metrics

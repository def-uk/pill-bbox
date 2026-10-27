from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

from data_process.data_yaml import data_yaml_path, load_class_map, yolo_dir
from data_process.make_labels import BBOX_COLS
from src.config import PROJECT_ROOT, get_device

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROCESSED_DIR / "results"
PREDICTIONS_DIR = PROCESSED_DIR / "predictions"
# 클래스 그룹: 라벨 버전 전체의 bbox 수 기준 (EDA 6.1, 9.3)
SIZE_GROUPS = [(21, "21+"), (6, "6-20"), (0, "≤5")]
# 일부만 바꿔 찍은 닮은 약 쌍 (EDA 8.2). 크기가 같은 원형 정제 46쌍은 GrabCut 측정이 필요해 뺐다.
CONFUSION_PAIRS = [(16262, 20238), (16232, 21325), (18357, 32310)]
PRED_COLS = ["file_name", "yolo_idx", "category_id", *BBOX_COLS, "conf"]


def size_group(n_bboxes: int) -> str:
    """bbox 수로 클래스 그룹 이름을 정한다."""
    return next(name for low, name in SIZE_GROUPS if n_bboxes >= low)


def evaluate_fold(
    weights: str | Path,
    label_version: str,
    split_name: str,
    fold: int,
    exp: str,
    imgsz: int = 640,
    batch: int = 16,
    conf: float = 0.001,
    iou: float = 0.7,
) -> tuple[pd.DataFrame, dict]:
    """fold의 val로 평가해 (클래스별 표, 요약)을 반환하고 results/<exp>/에 CSV로 저장한다.

    클래스별 표: yolo_idx, category_id, dl_name, 그룹, train/val bbox 수, AP50, AP50-95.
    val에 없는 클래스는 AP가 NaN이고 평균에서 빠진다. conf는 confusion matrix에도 쓰인다.
    iou는 NMS 기준이라 NMS-free 모델(YOLO26)에서는 영향이 없다.
    """
    class_map = load_class_map(label_version)
    labels = pd.read_csv(PROCESSED_DIR / "labels" / f"labels_{label_version}.csv")
    folds = pd.read_csv(PROCESSED_DIR / "splits" / split_name / f"fold_{fold}.csv")
    # train이 만든 runs/<exp>/<모델>_<분할>_fold<k>/weights/best.pt에서 이름을 가져온다.
    stem = Path(weights).parent.parent.name
    model = YOLO(weights)
    metrics = model.val(
        data=str(data_yaml_path(label_version, split_name, fold)),
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        device=get_device(),
        plots=True,  # confusion matrix는 plots=True일 때만 계산된다
        project=str(RESULTS_DIR / exp / "val"),
        name=stem,
    )
    per_class = class_map.copy()
    per_class["group"] = per_class["category_id"].map(
        labels["category_id"].value_counts().map(size_group)
    )
    split_of = labels["file_name"].map(folds.set_index("file_name")["split"])
    for split in ["train", "val"]:
        counts = labels.loc[split_of == split, "category_id"].value_counts()
        per_class[f"n_{split}_bbox"] = per_class["category_id"].map(counts).fillna(0).astype(int)
    box = metrics.box
    ap = pd.DataFrame({"ap50": box.ap50, "ap50_95": box.ap}, index=box.ap_class_index)
    per_class = per_class.join(ap, on="yolo_idx")
    per_class.loc[per_class["n_val_bbox"] == 0, ["ap50", "ap50_95"]] = np.nan

    matrix = metrics.confusion_matrix.matrix  # [예측, 정답], 마지막 행·열은 배경
    idx = dict(zip(class_map["category_id"], class_map["yolo_idx"], strict=True))
    pairs = {
        f"confuse_{a}_{b}": int(matrix[idx[a], idx[b]] + matrix[idx[b], idx[a]])
        for a, b in CONFUSION_PAIRS
    }
    summary = {
        "run": stem.rsplit("_fold", 1)[0],
        "weights": str(weights),
        "label_version": label_version,
        "split_name": split_name,
        "fold": fold,
        "map50": float(box.map50),
        "map50_95": float(box.map),
        **{
            f"ap50_95_{name}": float(per_class.loc[per_class["group"] == name, "ap50_95"].mean())
            for _, name in SIZE_GROUPS
        },
        "n_classes_not_in_val": int((per_class["n_val_bbox"] == 0).sum()),
        "n_val_only_classes": int(
            ((per_class["n_val_bbox"] > 0) & (per_class["n_train_bbox"] == 0)).sum()
        ),
        **pairs,
    }
    out_dir = RESULTS_DIR / exp
    out_dir.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(out_dir / f"{stem}_per_class.csv", index=False, encoding="utf-8")
    pd.DataFrame([summary]).to_csv(out_dir / f"{stem}_summary.csv", index=False, encoding="utf-8")
    return per_class, summary


def summarize_exp(exp: str) -> pd.DataFrame:
    """results/<exp>/의 fold 요약을 run(모델_분할)별 평균·표준편차로 모아 summary.csv로 저장한다."""
    out_dir = RESULTS_DIR / exp
    runs = pd.concat(
        [pd.read_csv(p) for p in sorted(out_dir.glob("*_summary.csv"))], ignore_index=True
    )
    numeric = runs.drop(columns=["fold"]).select_dtypes("number").columns
    summary = runs.groupby("run")[list(numeric)].agg(["mean", "std"])
    summary.insert(0, ("n_folds", "count"), runs.groupby("run")["fold"].nunique())
    summary.to_csv(out_dir / "summary.csv", encoding="utf-8")
    return summary


def predict_test(
    weights: str | Path,
    label_version: str,
    exp: str,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
) -> pd.DataFrame:
    """test 이미지를 예측해 category_id·COCO 픽셀 bbox로 되돌린 표를 test_pred.csv로 저장한다.

    입력은 이미지 경로뿐이고 메타데이터는 쓰지 않는다. 제출 형식은 아직 정해지지 않았다(#33).
    """
    class_map = load_class_map(label_version)
    to_category = dict(zip(class_map["yolo_idx"], class_map["category_id"], strict=True))
    test_dir = yolo_dir(label_version) / "images" / "test"
    model = YOLO(weights)
    rows = []
    for result in model.predict(
        source=str(test_dir),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=get_device(),
        stream=True,
        verbose=False,
    ):
        file_name = Path(result.path).name
        boxes = result.boxes
        for (x1, y1, x2, y2), cls, score in zip(
            boxes.xyxy.tolist(), boxes.cls.int().tolist(), boxes.conf.tolist(), strict=True
        ):
            rows.append([file_name, cls, to_category[cls], x1, y1, x2 - x1, y2 - y1, score])
    pred = pd.DataFrame(rows, columns=PRED_COLS)
    out_path = PREDICTIONS_DIR / exp / "test_pred.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_path, index=False, encoding="utf-8")
    return pred

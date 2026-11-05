import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from data_process.data_yaml import load_class_map, yolo_dir
from data_process.make_labels import BBOX_COLS
from src.config import RUNS_DIR, get_device

PRED_COLS = ["file_name", "yolo_idx", "category_id", *BBOX_COLS, "conf"]
# 대회 제출 csv 열. bbox는 원본 이미지 픽셀 기준 COCO 형식(왼쪽 위 x, y, 너비, 높이)
SUBMISSION_COLS = [
    "annotation_id",
    "image_id",
    "category_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "score",
]


def results_dir(exp: str, runs_dir: Path = RUNS_DIR) -> Path:
    """실험의 평가·예측 결과 폴더 <runs_dir>/<exp>/results를 반환한다."""
    return Path(runs_dir) / exp / "results"


def predict_boxes(
    model: YOLO,
    source: str | Path,
    label_version: str,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
) -> pd.DataFrame:
    """이미지를 예측해 PRED_COLS 표(category_id, 원본 픽셀 COCO bbox)로 반환한다.

    source는 이미지 폴더나 이미지 경로 목록 txt다.
    """
    class_map = load_class_map(label_version)
    to_category = dict(zip(class_map["yolo_idx"], class_map["category_id"], strict=True))
    rows = []
    for result in model.predict(
        source=str(source),
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
    return pd.DataFrame(rows, columns=PRED_COLS)


def predict_test(
    weights: str | Path,
    label_version: str,
    exp: str,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    runs_dir: Path = RUNS_DIR,
) -> pd.DataFrame:
    """test 이미지를 예측해 category_id·COCO 픽셀 bbox로 되돌린 표를 저장한다.

    저장 위치는 <runs_dir>/<exp>/results/<가중치 run 이름>_test_pred.csv.
    입력은 이미지 경로뿐이고 메타데이터는 쓰지 않는다. 제출 csv는 to_submission으로 바꾼다.
    """
    test_dir = yolo_dir(label_version) / "images" / "test"
    pred = predict_boxes(YOLO(weights), test_dir, label_version, imgsz, conf, iou)
    out_path = results_dir(exp, runs_dir) / f"{Path(weights).parent.parent.name}_test_pred.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_path, index=False, encoding="utf-8")
    return pred


def to_submission(pred: pd.DataFrame) -> pd.DataFrame:
    """predict_test 표를 대회 제출 형식(SUBMISSION_COLS)으로 바꾼다.

    image_id는 test 파일 이름의 숫자(12.png → 12)이고, annotation_id는 1부터 매긴 행 번호다.
    행은 image_id 순, 같은 사진 안에서는 score가 높은 순이다.
    """
    sub = pred.rename(
        columns={"x": "bbox_x", "y": "bbox_y", "w": "bbox_w", "h": "bbox_h", "conf": "score"}
    )
    sub["image_id"] = sub["file_name"].map(lambda name: int(Path(name).stem))
    sub = sub.sort_values(["image_id", "score"], ascending=[True, False], ignore_index=True)
    sub["annotation_id"] = sub.index + 1
    return sub[SUBMISSION_COLS]


def make_submission(
    weights: str | Path,
    label_version: str,
    exp: str,
    imgsz: int = 640,
    conf: float = 0.001,
    runs_dir: Path = RUNS_DIR,
) -> tuple[pd.DataFrame, Path]:
    """test 이미지를 예측해 제출 csv를 만들고 (제출 표, 저장 경로)를 반환한다.

    저장 위치는 <runs_dir>/<exp>/results/<가중치 run 이름>_submission_conf<conf>.csv.
    conf 미만의 박스는 버린다. 대회는 mAP 채점이라 점수 낮은 박스를 더해도 점수가 떨어지지
    않으므로 0.001을 기본으로 한다. 사진당 박스는 최대 300개다.
    """
    pred = predict_test(weights, label_version, exp, imgsz, conf, runs_dir=runs_dir)
    sub = to_submission(pred)
    stem = Path(weights).parent.parent.name
    out_path = results_dir(exp, runs_dir) / f"{stem}_submission_conf{conf:g}.csv"
    sub.to_csv(out_path, index=False, encoding="utf-8")
    return sub, out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="test 예측과 제출 csv 만들기")
    parser.add_argument("--weights", type=Path, required=True, help="best.pt 경로")
    parser.add_argument("--exp", required=True, help="결과를 둘 실험 이름")
    parser.add_argument("--label", default="v1", help="학습에 쓴 라벨 버전")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001, help="이 점수 미만의 박스는 버림")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args()
    sub, out_path = make_submission(
        args.weights, args.label, args.exp, args.imgsz, args.conf, args.runs_dir
    )
    print(f"사진 {sub['image_id'].nunique()}장, 박스 {len(sub)}개: {out_path}")


if __name__ == "__main__":
    main()

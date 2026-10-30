import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from matplotlib.figure import Figure
from ultralytics import YOLO

from data_process.data_yaml import data_yaml_path, load_class_map, yolo_dir
from data_process.make_labels import BBOX_COLS
from src.config import PROJECT_ROOT, RUNS_DIR, get_device
from visualization import plot_fold_metrics, plot_per_class_ap, save_figure, set_korean_font

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
# 주 지표: IoU 0.75–0.95(0.05 간격 5단계) AP 평균. 박스가 알약에 꼭 맞는지를 엄하게 본다.
# ultralytics all_ap의 10열은 IoU 0.50–0.95라 5열부터 쓴다. 대회 채점 지표 mAP@[0.75:0.95]와 같다.
MAIN_METRIC = "map75_95"
IOU75_COL = 5
# 클래스 그룹: 라벨 버전 전체의 bbox 수 기준 (EDA 6.1, 9.3)
SIZE_GROUPS = [(21, "21+"), (6, "6-20"), (0, "≤5")]
# 일부만 바꿔 찍은 닮은 약 쌍 (EDA 8.2). 크기가 같은 원형 정제 46쌍은 GrabCut 측정이 필요해 뺐다.
CONFUSION_PAIRS = [(16262, 20238), (16232, 21325), (18357, 32310)]
PRED_COLS = ["file_name", "yolo_idx", "category_id", *BBOX_COLS, "conf"]
# 정밀도·재현율·평균 IoU의 계산 기준 (평가지표 선정안 #33). 신뢰도 0.5 이상인 예측만 답으로 보고,
# 정답과 IoU 0.5 이상 겹치면 같은 알약으로 짝짓는다. ultralytics의 정밀도·재현율은 실험마다
# F1이 최대인 신뢰도에서 계산해 서로 비교할 수 없으므로 고정 기준으로 따로 센다.
POINT_CONF = 0.5
POINT_IOU = 0.5
# 추론 시간: ultralytics 검증이 잰 사진 한 장당 단계별 시간(ms). 같은 장비·배치 크기에서 잰 값끼리만
# 비교한다. postprocess에 NMS가 들어 있어 NMS 유무의 속도 차이는 이 열에서 보인다.
SPEED_KEYS = ["preprocess", "inference", "postprocess"]
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


def ap75_95(all_ap: np.ndarray) -> np.ndarray:
    """클래스 × IoU 10단계 AP 표에서 클래스별 AP75-95를 구한다."""
    return all_ap[:, IOU75_COL:].mean(1)


def map75_95(box: object) -> float:
    """ultralytics 검증 결과(metrics.box)의 mAP75-95. 학습 중 best.pt 선택에도 쓴다."""
    return float(ap75_95(box.all_ap).mean()) if len(box.all_ap) else 0.0


def results_dir(exp: str, runs_dir: Path = RUNS_DIR) -> Path:
    """실험의 평가·예측 결과 폴더 <runs_dir>/<exp>/results를 반환한다."""
    return Path(runs_dir) / exp / "results"


def size_group(n_bboxes: int) -> str:
    """bbox 수로 클래스 그룹 이름을 정한다."""
    return next(name for low, name in SIZE_GROUPS if n_bboxes >= low)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[x, y, w, h] 박스 a(n개)와 b(m개)의 IoU 표(n × m)를 구한다."""
    inter_w = np.minimum(a[:, None, 0] + a[:, None, 2], b[None, :, 0] + b[None, :, 2])
    inter_w = (inter_w - np.maximum(a[:, None, 0], b[None, :, 0])).clip(0)
    inter_h = np.minimum(a[:, None, 1] + a[:, None, 3], b[None, :, 1] + b[None, :, 3])
    inter_h = (inter_h - np.maximum(a[:, None, 1], b[None, :, 1])).clip(0)
    inter = inter_w * inter_h
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter
    return inter / union


def match_boxes(pred: pd.DataFrame, gt: pd.DataFrame, iou_thres: float = POINT_IOU) -> pd.DataFrame:
    """사진마다 신뢰도가 높은 예측부터, 아직 짝이 없는 정답 중 IoU가 가장 큰 것과 짝짓는다.

    클래스와 관계없이 위치로만 짝짓고(IoU ≥ iou_thres), 짝마다 예측·정답 category_id와 IoU를
    반환한다. 클래스가 맞는지는 짝을 지은 뒤에 본다.
    """
    rows = []
    gt_by_file = dict(tuple(gt.groupby("file_name")))
    pred = pred.sort_values("conf", ascending=False)
    for file_name, p in pred.groupby("file_name", sort=False):
        g = gt_by_file.get(file_name)
        if g is None:
            continue
        iou = box_iou(p[BBOX_COLS].to_numpy(float), g[BBOX_COLS].to_numpy(float))
        free = np.ones(len(g), dtype=bool)
        for i, pred_category in enumerate(p["category_id"]):
            j = int(np.where(free, iou[i], -1).argmax())
            if iou[i, j] >= iou_thres and free[j]:
                free[j] = False
                rows.append([file_name, pred_category, g["category_id"].iat[j], iou[i, j]])
    return pd.DataFrame(rows, columns=["file_name", "pred_category", "gt_category", "iou"])


def point_metrics(
    pred: pd.DataFrame, gt: pd.DataFrame, conf: float = POINT_CONF, iou_thres: float = POINT_IOU
) -> dict[str, float]:
    """신뢰도 conf 이상인 예측으로 정밀도·재현율·F1·평균 IoU를 센다. 짝은 match_boxes로 짓는다.

    정밀도 = 약까지 맞힌 짝 / 예측 수, 재현율 = 약까지 맞힌 짝 / 정답 수,
    F1 = 2 × 정밀도 × 재현율 / (정밀도 + 재현율),
    평균 IoU = 짝지은 쌍 전체(다른 약으로 답한 짝 포함)의 IoU 평균,
    n_wrong_class = 다른 약으로 답한 짝 수.
    """
    pred = pred[pred["conf"] >= conf]
    pairs = match_boxes(pred, gt, iou_thres)
    n_correct = int((pairs["pred_category"] == pairs["gt_category"]).sum())
    precision = n_correct / len(pred) if len(pred) else np.nan
    recall = n_correct / len(gt) if len(gt) else np.nan
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_iou": float(pairs["iou"].mean()) if len(pairs) else np.nan,
        "n_wrong_class": len(pairs) - n_correct,
    }


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


def box_metrics(metrics: object) -> dict[str, float]:
    """ultralytics 검증 결과에서 mAP 네 가지와 사진 한 장당 단계별 추론 시간(ms)을 꺼낸다."""
    box = metrics.box
    return {
        MAIN_METRIC: map75_95(box),
        "map50": float(box.map50),
        "map75": float(box.map75),
        "map50_95": float(box.map),
        **{f"ms_{key}": float(metrics.speed[key]) for key in SPEED_KEYS},
    }


def supports_nms_free(weights: str | Path) -> bool:
    """가중치에 NMS 없이 예측하는 one2one 출력층이 있는지 본다(YOLO26은 있음, YOLO11은 없음)."""
    return getattr(YOLO(weights).model.model[-1], "one2one_cv2", None) is not None


def evaluate_nms_free(
    weights: str | Path, data: Path, out_dir: Path, imgsz: int, batch: int, conf: float
) -> dict[str, float]:
    """one2one 출력층으로 NMS 없이 검증해 box_metrics를 반환한다.

    ultralytics 기본값(nms=None)은 YOLO26도 one2many 출력층 + NMS로 예측하므로 따로 잰다.
    앞선 검증의 fuse가 쓰지 않는 출력층을 지우므로 가중치를 새로 읽는다.
    """
    metrics = YOLO(weights).val(
        data=str(data),
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        nms=False,
        device=get_device(),
        plots=False,
        project=str(out_dir / "val"),
        name=f"{Path(weights).parent.parent.name}_nms_free",
        exist_ok=True,
    )
    return box_metrics(metrics)


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
    runs_dir: Path = RUNS_DIR,
) -> tuple[pd.DataFrame, dict]:
    """fold의 val로 평가해 (클래스별 표, 요약)을 반환하고 <runs_dir>/<exp>/results/에 저장한다.

    클래스별 표: yolo_idx, category_id, dl_name, 그룹, train/val bbox 수, AP50, AP50-95, AP75-95.
    val에 없는 클래스는 AP가 NaN이고 평균에서 빠진다. conf는 confusion matrix에도 쓰인다.
    iou는 NMS 기준이다. YOLO26도 ultralytics 기본값에서는 NMS를 거쳐 평가되므로 영향을 받는다.
    요약의 precision·recall·f1·mean_iou·n_wrong_class는 val을 따로 예측해 고정 기준
    (POINT_CONF, POINT_IOU)으로 센 값이다(point_metrics). ms_*는 사진 한 장당 추론 시간이다.
    one2one 출력층이 있는 모델(YOLO26)은 NMS 없이도 검증해 NMS 사용 점수와 함께
    <stem>_nms_free.csv(head 열: nms / nms_free)에 따로 저장한다.
    """
    class_map = load_class_map(label_version)
    labels = pd.read_csv(PROCESSED_DIR / "labels" / f"labels_{label_version}.csv")
    folds = pd.read_csv(PROCESSED_DIR / "splits" / split_name / f"fold_{fold}.csv")
    out_dir = results_dir(exp, runs_dir)
    # train이 만든 <runs_dir>/<exp>/<run 이름>/weights/best.pt에서 run 이름을 가져온다.
    stem = Path(weights).parent.parent.name
    data = data_yaml_path(label_version, split_name, fold)
    # 검증의 fuse가 one2one 출력층을 지우기 전에 확인한다
    nms_free = supports_nms_free(weights)
    model = YOLO(weights)
    metrics = model.val(
        data=str(data),
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        device=get_device(),
        plots=True,  # confusion matrix는 plots=True일 때만 계산된다
        project=str(out_dir / "val"),
        name=stem,
        exist_ok=True,
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
    ap_cols = ["ap50", "ap50_95", "ap75_95"]
    ap = pd.DataFrame(
        {"ap50": box.ap50, "ap50_95": box.ap, "ap75_95": ap75_95(box.all_ap)},
        index=box.ap_class_index,
    )
    per_class = per_class.join(ap, on="yolo_idx")
    per_class.loc[per_class["n_val_bbox"] == 0, ap_cols] = np.nan

    matrix = metrics.confusion_matrix.matrix  # [예측, 정답], 마지막 행·열은 배경
    idx = dict(zip(class_map["category_id"], class_map["yolo_idx"], strict=True))
    pairs = {
        f"confuse_{a}_{b}": int(matrix[idx[a], idx[b]] + matrix[idx[b], idx[a]])
        for a, b in CONFUSION_PAIRS
    }
    val_list = yaml.safe_load(data.read_text(encoding="utf-8"))["val"]
    pred = predict_boxes(model, val_list, label_version, imgsz, conf=POINT_CONF, iou=iou)
    info = {
        "run": stem.rsplit("_fold", 1)[0],
        "weights": str(weights),
        "label_version": label_version,
        "split_name": split_name,
        "fold": fold,
    }
    metric = box_metrics(metrics)
    summary = {
        **info,
        **metric,
        **point_metrics(pred, labels[split_of == "val"]),
        **{
            f"ap75_95_{name}": float(per_class.loc[per_class["group"] == name, "ap75_95"].mean())
            for _, name in SIZE_GROUPS
        },
        "n_classes_not_in_val": int((per_class["n_val_bbox"] == 0).sum()),
        "n_val_only_classes": int(
            ((per_class["n_val_bbox"] > 0) & (per_class["n_train_bbox"] == 0)).sum()
        ),
        **pairs,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(out_dir / f"{stem}_per_class.csv", index=False, encoding="utf-8")
    pd.DataFrame([summary]).to_csv(out_dir / f"{stem}_summary.csv", index=False, encoding="utf-8")
    if nms_free:
        free = evaluate_nms_free(weights, data, out_dir, imgsz, batch, conf)
        heads = pd.DataFrame(
            [{**info, "head": "nms", **metric}, {**info, "head": "nms_free", **free}]
        )
        heads.to_csv(out_dir / f"{stem}_nms_free.csv", index=False, encoding="utf-8")
    return per_class, summary


def summarize_exp(exp: str, runs_dir: Path = RUNS_DIR) -> pd.DataFrame:
    """fold 요약을 run(모델_분할_증강)별 평균·표준편차로 모아 summary.csv에 저장한다.

    행은 주 지표(MAIN_METRIC) 평균이 높은 순이다. NMS-free 결과(*_nms_free.csv)가 있으면
    run·head별 평균·표준편차를 summary_nms_free.csv에 따로 저장한다.
    """
    out_dir = results_dir(exp, runs_dir)
    runs = pd.concat(
        [pd.read_csv(p) for p in sorted(out_dir.glob("*_summary.csv"))], ignore_index=True
    )
    numeric = runs.drop(columns=["fold"]).select_dtypes("number").columns
    summary = runs.groupby("run")[list(numeric)].agg(["mean", "std"])
    summary.insert(0, ("n_folds", "count"), runs.groupby("run")["fold"].nunique())
    summary = summary.sort_values((MAIN_METRIC, "mean"), ascending=False)
    summary.to_csv(out_dir / "summary.csv", encoding="utf-8")
    heads = [pd.read_csv(p) for p in sorted(out_dir.glob("*_nms_free.csv"))]
    if heads:
        heads = pd.concat(heads, ignore_index=True)
        numeric = heads.drop(columns=["fold"]).select_dtypes("number").columns
        by_head = heads.groupby(["run", "head"])[list(numeric)].agg(["mean", "std"])
        by_head.to_csv(out_dir / "summary_nms_free.csv", encoding="utf-8")
    return summary


def load_fold_results(
    exps: Sequence[str], runs_dir: Path = RUNS_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """실험들의 fold 요약과 클래스별 표를 모아 (요약, 클래스별)로 반환한다.

    evaluate_fold가 저장한 <runs_dir>/<exp>/results/*_summary.csv와 *_per_class.csv를 읽는다.
    두 표에 exp, run, fold 열을 붙이고, 클래스별 표에는 전체 bbox 수 n_bbox를 더한다.
    """
    summaries, per_class = [], []
    for exp in exps:
        out_dir = results_dir(exp, runs_dir)
        for path in sorted(out_dir.glob("*_summary.csv")):
            summary = pd.read_csv(path).assign(exp=exp)
            table = pd.read_csv(out_dir / path.name.replace("_summary.csv", "_per_class.csv"))
            summaries.append(summary)
            per_class.append(
                table.assign(exp=exp, run=summary.at[0, "run"], fold=summary.at[0, "fold"])
            )
    if not summaries:
        raise FileNotFoundError(f"{exps}의 평가 결과가 없습니다. evaluate_fold를 먼저 실행하세요.")
    per_class = pd.concat(per_class, ignore_index=True)
    per_class["n_bbox"] = per_class["n_train_bbox"] + per_class["n_val_bbox"]
    return pd.concat(summaries, ignore_index=True), per_class


def plot_results(
    exps: str | Sequence[str], runs_dir: Path = RUNS_DIR, save_dir: Path | None = None
) -> dict[str, Figure]:
    """실험들의 fold 평가 결과를 그림 네 장으로 그려 {이름: 그림}으로 반환한다.

    - fold_metrics: run별 mAP75-95와 클래스 그룹별 AP75-95 (fold 점수, 평균 ± 표준편차)
    - point_metrics: 고정 기준(POINT_CONF, POINT_IOU)의 정밀도·재현율·F1·평균 IoU
    - confusion: 닮은 약 쌍(CONFUSION_PAIRS)이 서로 헷갈린 횟수. 낮을수록 좋다
    - per_class: 클래스별 AP75-95 (fold 평균, fold 간 범위). 낮은 클래스가 위에 온다
    여러 실험을 넘기면 한 그림에서 비교하고, run 이름표 앞에 실험 이름을 붙인다.
    save_dir을 주면 <save_dir>/<이름>.png로도 저장한다.
    """
    exps = [exps] if isinstance(exps, str) else list(exps)
    set_korean_font()
    summaries, per_class = load_fold_results(exps, runs_dir)
    # 분할이 하나뿐이면 run 이름표에서 분할 이름을 빼고 제목에 한 번만 적는다
    splits = summaries["split_name"].unique()
    for table in (summaries, per_class):
        label = table["run"]
        if len(splits) == 1:
            label = label.str.replace(f"_{splits[0]}", "", regex=False)
        table["label"] = table["exp"] + "/" + label if len(exps) > 1 else label
    # 모든 그림에서 run을 주 지표 평균이 높은 순으로 놓는다
    order = summaries.groupby("label")[MAIN_METRIC].mean().sort_values(ascending=False).index
    scope = f"{', '.join(exps)} · {', '.join(splits)}"
    groups = [name for _, name in SIZE_GROUPS]
    names = load_class_map(summaries.at[0, "label_version"]).set_index("category_id")["dl_name"]
    names = names.str.strip()
    ap_metrics = {MAIN_METRIC: "mAP75-95"} | {
        f"ap75_95_{g}": f"AP75-95 · bbox {g} 클래스" for g in groups
    }
    point = {"precision": "정밀도", "recall": "재현율", "f1": "F1", "mean_iou": "평균 IoU"}
    confusion = {f"confuse_{a}_{b}": f"{names[a]}\n↔ {names[b]}" for a, b in CONFUSION_PAIRS}
    # 이 지표가 생기기 전에 평가한 결과에는 열이 없으므로 있는 열만 그린다
    point = {col: name for col, name in point.items() if col in summaries}
    confusion = {col: name for col, name in confusion.items() if col in summaries}
    point_title = (
        f"conf {POINT_CONF:g} · IoU {POINT_IOU:g} 기준 정밀도·재현율·F1·평균 IoU ({scope})"
    )
    figures = {
        "fold_metrics": plot_fold_metrics(
            summaries, ap_metrics, title=f"fold 점수 ({scope})", order=order
        ),
    }
    if point:
        figures["point_metrics"] = plot_fold_metrics(
            summaries, point, title=point_title, order=order
        )
    if confusion:
        figures["confusion"] = plot_fold_metrics(
            summaries,
            confusion,
            title=f"닮은 약 쌍 혼동 횟수, 낮을수록 좋음 ({scope})",
            order=order,
            counts=True,
        )
    figures["per_class"] = plot_per_class_ap(
        per_class, groups, title=f"클래스별 AP75-95 (괄호는 bbox 수, {scope})", order=order
    )
    if save_dir is not None:
        for name, fig in figures.items():
            save_figure(fig, Path(save_dir) / f"{name}.png", dpi=120)
    return figures


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
    않으므로, 검증(evaluate_fold)과 같은 0.001을 기본으로 한다. 사진당 박스는 최대 300개다.
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

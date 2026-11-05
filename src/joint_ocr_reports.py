"""공동 모델의 에폭별 곡선·주/보조 지표·클래스별 AP·혼동행렬을 저장한다."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from ultralytics.data.utils import img2label_paths
from ultralytics.utils.metrics import ap_per_class

from models.evaluate import CONFUSION_PAIRS, box_iou, match_boxes, point_metrics, size_group
from models.ocr import evaluate_ocr_predictions
from src.ocr_reports import plot_detection_confusions, plot_ocr_comparison


def class_ap(
    predictions: pd.DataFrame, truth: pd.DataFrame, class_map: pd.DataFrame
) -> pd.DataFrame:
    """검증 정답이 없는 클래스는 NaN으로 두고 클래스별 AP를 계산한다."""
    correct, scores, classes = [], [], []
    thresholds = np.linspace(0.5, 0.95, 10)
    for name, targets in truth.groupby("file_name"):
        detected = predictions.loc[predictions["file_name"].eq(name)].sort_values(
            "conf", ascending=False
        )
        overlap = box_iou(
            targets[["x", "y", "w", "h"]].to_numpy(float),
            detected[["x", "y", "w", "h"]].to_numpy(float),
        )
        overlap *= (
            targets["category_id"].to_numpy()[:, None]
            == detected["category_id"].to_numpy()[None, :]
        )
        matches = np.zeros((len(detected), 10), dtype=bool)
        claimed = np.zeros((len(targets), 10), dtype=bool)
        for index in range(len(detected)):
            available = np.where(claimed, 0.0, overlap[:, index, None])
            best = available.argmax(axis=0)
            matches[index] = available[best, np.arange(10)] >= thresholds
            claimed[best, np.arange(10)] |= matches[index]
        correct.append(matches)
        scores.extend(detected["conf"])
        classes.extend(detected["category_id"])
    stats = ap_per_class(
        np.concatenate(correct),
        np.asarray(scores),
        np.asarray(classes),
        truth["category_id"].to_numpy(),
    )
    ap = stats[5]
    table = pd.DataFrame(
        {
            "category_id": stats[6].astype(int),
            "ap50": ap[:, 0],
            "ap50_95": ap.mean(1),
            "ap75_95": ap[:, 5:].mean(1),
        }
    )
    result = class_map.merge(table, on="category_id", how="left")
    result["n_val_bbox"] = (
        result["category_id"].map(truth["category_id"].value_counts()).fillna(0).astype(int)
    )
    return result


def plot_joint_history(history_path: Path, loss_weight: float) -> Figure:
    """실제로 기록한 매 에폭 검출·CTC·융합 성능을 한 그림에 표시한다."""
    history = pd.read_csv(history_path)
    history.columns = history.columns.str.strip()
    figure, axes = plt.subplots(2, 3, figsize=(17, 9))
    for split in ["train", "val"]:
        detection = history[[f"{split}/{name}_loss" for name in ["box", "cls", "dfl"]]].sum(axis=1)
        ctc = history[f"{split}/ocr_loss"]
        axes[0, 0].plot(history["epoch"], detection, label=f"{split} detection")
        axes[0, 1].plot(history["epoch"], ctc, label=f"{split} CTC")
        axes[0, 2].plot(history["epoch"], detection + loss_weight * ctc, label=f"{split} total")
    for column, label in [("metrics/mAP75-95(B)", "YOLO head"), ("joint/map75_95", "YOLO + OCR")]:
        axes[1, 0].plot(history["epoch"], history[column], label=label)
    axes[1, 1].plot(history["epoch"], history["ocr/cer"], label="OCR CER (GT ROI)")
    axes[1, 2].plot(
        history["epoch"], history["ocr/exact_accuracy"], label="OCR exact accuracy (GT ROI)"
    )
    titles = [
        "Detection loss",
        "OCR CTC loss",
        f"Total loss = detection + {loss_weight:g} * CTC",
        "Same-validation mAP75-95",
        "OCR character error rate",
        "OCR string accuracy",
    ]
    for ax, title in zip(axes.flat, titles, strict=True):
        ax.set(title=title, xlabel="Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(history_path.parent / "joint_epoch_curves.png", dpi=160)
    return figure


def save_joint_report(
    prediction_dir: Path,
    spec: dict,
    *,
    point_conf: float,
    point_iou: float,
    cm_conf: float,
    cm_iou: float,
) -> tuple[dict[str, pd.DataFrame], list[Figure]]:
    """같은 검증 예측에 주 지표와 고정 기준 보조 지표·혼동행렬을 만든다."""
    truth = pd.read_csv(prediction_dir / "validation_truth.csv")
    predictions = {
        "YOLO": pd.read_csv(prediction_dir / "yolo_predictions.csv"),
        "YOLO + OCR": pd.read_csv(prediction_dir / "ocr_predictions.csv"),
    }
    class_map = pd.DataFrame(spec["class_map"])
    summary, auxiliary, per_class = {}, {}, []
    counts = {}
    for path in img2label_paths(spec["inventory"]["train"]):
        for line in Path(path).read_text().splitlines():
            if line.strip():
                index = int(line.split()[0])
                counts[index] = counts.get(index, 0) + 1
    for name, table in predictions.items():
        summary[name] = evaluate_ocr_predictions(table, truth)
        points = point_metrics(table, truth, conf=point_conf, iou_thres=point_iou)
        pairs = match_boxes(table.loc[table["conf"] >= cm_conf], truth, cm_iou)
        for first, second in CONFUSION_PAIRS:
            points[f"confuse_{first}_{second}"] = int(
                (
                    (pairs["pred_category"].eq(first) & pairs["gt_category"].eq(second))
                    | (pairs["pred_category"].eq(second) & pairs["gt_category"].eq(first))
                ).sum()
            )
        auxiliary[name] = points
        classes = class_ap(table, truth, class_map)
        classes["n_train_bbox"] = classes["yolo_idx"].map(counts).fillna(0).astype(int)
        classes["group"] = (classes["n_train_bbox"] + classes["n_val_bbox"]).map(size_group)
        classes["model"] = name
        for group in classes["group"].unique():
            summary[name][f"ap75_95_group_{group}"] = classes.loc[
                classes["group"].eq(group), "ap75_95"
            ].mean()
        per_class.append(classes)
    summary = pd.DataFrame(summary).T
    auxiliary = pd.DataFrame(auxiliary).T
    classes = pd.concat(per_class, ignore_index=True)
    tables = {
        "validation_comparison": summary,
        "auxiliary_metrics": auxiliary,
        "per_class_metrics": classes,
    }
    for name, table in tables.items():
        table.to_csv(prediction_dir / f"{name}.csv", index=name != "per_class_metrics")
    figures = [plot_ocr_comparison(summary, prediction_dir)]
    figure, ax = plt.subplots(figsize=(11, 5))
    auxiliary[["precision", "recall", "f1", "mean_iou"]].T.plot.bar(ax=ax, rot=0, ylim=(0, 1.15))
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=3)
    ax.set(
        title=f"Fixed-threshold auxiliary metrics: conf >= {point_conf:g}, IoU >= {point_iou:g}",
        ylabel="Score",
    )
    figure.tight_layout()
    figure.savefig(prediction_dir / "auxiliary_metrics.png", dpi=160)
    figures.append(figure)
    for columns, filename, title in [
        (
            [column for column in summary if column.startswith("ap75_95_group_")],
            "class_group_ap75_95",
            "AP75-95 by class support (train + val bbox counts)",
        ),
        (
            [column for column in auxiliary if column.startswith("confuse_")],
            "similar_pill_confusions",
            f"Similar-pill confusion counts: conf >= {cm_conf:g}, IoU >= {cm_iou:g}",
        ),
    ]:
        table = summary if filename == "class_group_ap75_95" else auxiliary
        figure, ax = plt.subplots(figsize=(12, 5))
        table[columns].T.plot.bar(ax=ax, rot=15)
        ax.set(title=title, ylabel="AP75-95" if filename == "class_group_ap75_95" else "Count")
        ax.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(prediction_dir / f"{filename}.png", dpi=160)
        figures.append(figure)
    figure, ax = plt.subplots(figsize=(12, max(5, class_map.shape[0] * 0.22)))
    shown = classes.pivot(index="category_id", columns="model", values="ap75_95").dropna(how="all")
    shown.sort_values("YOLO + OCR").plot.barh(ax=ax, xlim=(0, 1.1))
    ax.set(
        title="Class AP75-95 (classes without validation GT are omitted)",
        xlabel="AP75-95",
        ylabel="category_id",
    )
    figure.tight_layout()
    figure.savefig(prediction_dir / "per_class_ap75_95.png", dpi=160)
    figures.append(figure)
    figures.extend(
        plot_detection_confusions(
            predictions, truth, class_map, prediction_dir, conf=cm_conf, iou=cm_iou
        )
    )
    return tables, figures

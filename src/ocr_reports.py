"""독립 OCR 학습과 YOLO 재분류 평가 결과를 표와 그래프로 정리한다."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def plot_yolo_ocr_report(
    yolo_history_path: Path, ocr_history_path: Path, comparison_path: Path
) -> list[Figure]:
    """각 모델의 에폭별 학습 곡선과 최종 융합 평가를 함께 저장해 반환한다."""
    history = pd.read_csv(yolo_history_path)
    history.columns = history.columns.str.strip()
    figure, axes = plt.subplots(1, 3, figsize=(16, 4))
    for column in ["train/box_loss", "train/cls_loss", "train/dfl_loss"]:
        axes[0].plot(history["epoch"], history[column], label=column)
    for column in ["val/box_loss", "val/cls_loss", "val/dfl_loss"]:
        axes[1].plot(history["epoch"], history[column], label=column)
    for column in ["metrics/mAP75-95(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
        axes[2].plot(history["epoch"], history[column], label=column)
    for ax, title in zip(
        axes, ["YOLO train loss", "YOLO validation loss", "YOLO validation mAP"], strict=True
    ):
        ax.set(title=title, xlabel="Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[2].set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(yolo_history_path.parent / "yolo_epoch_curves.png", dpi=160)
    # 최종 융합 평가는 학습 이력과 별도 그림으로 표시해 에폭별 기록으로 오해하지 않게 한다.
    comparison = pd.read_csv(comparison_path, index_col=0)
    return [
        figure,
        plot_ocr_history(ocr_history_path),
        plot_ocr_comparison(comparison, comparison_path.parent),
    ]


def plot_ocr_history(history_path: Path) -> Figure:
    """에폭별 OCR 학습 손실·검증 CER·문자열 정확도 그래프를 저장한다."""
    history = pd.read_csv(history_path)
    if history.empty:
        raise ValueError("OCR 학습 이력이 비어 있습니다.")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, column, title in zip(
        axes,
        ["train_loss", "val_cer", "val_exact_accuracy"],
        ["OCR training loss", "OCR validation CER (lower is better)", "OCR exact accuracy"],
        strict=True,
    ):
        ax.plot(history["epoch"], history[column], marker=".", linewidth=1.5)
        ax.set(title=title, xlabel="Epoch", ylabel=column)
        ax.grid(alpha=0.3)
    axes[2].set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(history_path.parent / "ocr_training_curves.png", dpi=160, bbox_inches="tight")
    return figure


def plot_ocr_comparison(metrics: pd.DataFrame, output_dir: Path) -> Figure:
    """동일 validation에서 구한 YOLO 단독·OCR 재분류 지표를 그래프로 저장한다."""
    required = ["map75_95", "map50", "map50_95", "precision", "recall"]
    models = ["YOLO", "YOLO + OCR"]
    if set(required) - set(metrics.columns) or set(models) - set(metrics.index):
        raise ValueError("YOLO·OCR 비교 표에 필수 평가지표가 없습니다.")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=(12, 6))
    table = metrics.loc[models, required].T.copy()
    table.index = ["mAP75-95\n(main metric)", "mAP50", "mAP50-95", "Precision", "Recall"]
    table.plot.bar(ax=ax, color=["steelblue", "darkorange"], rot=0, width=0.75)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=3, fontsize=9)
    ax.set(
        title="Same validation images: YOLO vs YOLO + OCR",
        ylabel="Score",
        xlabel="",
        ylim=(0, 1.15),
    )
    ax.legend(title="Prediction", loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(output_dir / "validation_comparison.png", dpi=160, bbox_inches="tight")
    return figure


def plot_detection_confusions(
    predictions: dict[str, pd.DataFrame],
    truth: pd.DataFrame,
    class_map: pd.DataFrame,
    output_dir: Path,
    *,
    conf: float,
    iou: float,
) -> list[Figure]:
    """동일 검증 이미지의 검출 혼동행렬과 정답별 정규화 행렬을 저장한다."""
    import numpy as np
    import torch
    from ultralytics.utils.metrics import ConfusionMatrix

    output_dir.mkdir(parents=True, exist_ok=True)
    classes = class_map.sort_values("yolo_idx")
    to_index = dict(zip(classes["category_id"], range(len(classes)), strict=True))
    labels = classes["category_id"].astype(str).tolist() + ["background"]
    classes.to_csv(output_dir / "confusion_class_map.csv", index=False)
    matrices = {}
    for name, table in predictions.items():
        matrix = ConfusionMatrix(names=dict(enumerate(labels[:-1])))
        for file_name in sorted(set(truth["file_name"]) | set(table["file_name"])):
            targets = truth.loc[truth["file_name"].eq(file_name)]
            detected = table.loc[table["file_name"].eq(file_name)]
            batches = []
            for rows in [targets, detected]:
                boxes = rows[["x", "y", "w", "h"]].to_numpy(dtype=float, copy=True)
                boxes[:, 2:] += boxes[:, :2]
                batches.append(
                    {
                        "bboxes": torch.tensor(boxes, dtype=torch.float32),
                        "cls": torch.tensor(
                            rows["category_id"].map(to_index).to_numpy(), dtype=torch.int64
                        ),
                    }
                )
            batches[1]["conf"] = torch.tensor(detected["conf"].to_numpy(), dtype=torch.float32)
            matrix.process_batch(batches[1], batches[0], conf=conf, iou_thres=iou)
        matrices[name] = matrix.matrix
        stem = "yolo" if name == "YOLO" else "yolo_ocr"
        pd.DataFrame(matrix.matrix, index=labels, columns=labels).to_csv(
            output_dir / f"confusion_{stem}.csv"
        )
    figures = []
    for normalized in [False, True]:
        figure, axes = plt.subplots(1, len(matrices), figsize=(24, 12), squeeze=False)
        for ax, (name, counts) in zip(axes.flat, matrices.items(), strict=True):
            values = (
                counts / np.maximum(counts.sum(axis=0, keepdims=True), 1) if normalized else counts
            )
            shown = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized else None)
            ax.set(
                title=f"{name}: conf > {conf:g}, matching IoU > {iou:g}",
                xlabel="True category",
                ylabel="Predicted category",
            )
            ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
            ax.set_yticks(range(len(labels)), labels, fontsize=6)
            for row, column in np.argwhere(values > 0):
                value = values[row, column]
                ax.text(
                    column,
                    row,
                    f"{value:.2f}" if normalized else f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=4,
                    color="white" if value > values.max() / 2 else "black",
                )
            figure.colorbar(shown, ax=ax, fraction=0.035)
        suffix = "normalized" if normalized else "counts"
        figure.suptitle(f"Detection confusion matrices ({suffix}); background = missed / extra box")
        figure.tight_layout()
        figure.savefig(output_dir / f"confusion_comparison_{suffix}.png", dpi=160)
        figures.append(figure)
    return figures

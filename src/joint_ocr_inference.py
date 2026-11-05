"""공동 체크포인트 하나로 검출·OCR 보정·전체 클래스 제출 CSV를 만든다."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.data.utils import img2label_paths
from ultralytics.utils import nms, ops

from models.evaluate import PRED_COLS, to_submission
from models.joint_ocr import (
    JointDetectionModel,
    prediction_rows,
    revise_predictions,
    shared_feature,
)
from src.config import get_device


@torch.inference_mode()
def predict_joint_checkpoint(
    weights: Path,
    images: list[Path],
    output_root: Path,
    *,
    split: str,
    imgsz: int,
    conf: float,
    nms_iou: float,
    max_det: int,
) -> Path:
    """이미지 한 장씩 공동 모델로 예측해 단독·융합·근거·정답 표를 저장한다."""
    if split not in {"val", "test"} or not images:
        raise ValueError("공동 예측은 비어 있지 않은 val/test 이미지 목록이 필요합니다.")
    model = YOLO(weights).model.float().to(get_device()).eval()
    if not isinstance(model, JointDetectionModel):
        raise TypeError("독립 OCR 모델 대신 공동 학습의 best.pt를 지정하세요.")
    spec = model.joint_spec
    original_rows, revised_rows, truth_rows, audit = [], [], [], []
    device = next(model.parameters()).device
    for path in tqdm(images, desc=f"공동 모델 {split}", mininterval=5):
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
        height, width = pixels.shape[:2]
        letterboxed = LetterBox(new_shape=(imgsz, imgsz), auto=False)(image=pixels)
        tensor = torch.from_numpy(np.ascontiguousarray(letterboxed[:, :, ::-1].transpose(2, 0, 1)))
        tensor = tensor.to(device).float()[None] / 255
        raw = model(tensor)
        outputs = nms.non_max_suppression(raw, conf, nms_iou, multi_label=True, max_det=max_det)[0]
        prediction = {"bboxes": outputs[:, :4], "conf": outputs[:, 4], "cls": outputs[:, 5]}
        revised, changes = revise_predictions(
            model, shared_feature(raw), [prediction], tensor.shape[2:]
        )
        for output, rows in [(prediction, original_rows), (revised[0], revised_rows)]:
            output["bboxes"] = ops.scale_boxes(
                tensor.shape[2:], output["bboxes"].clone(), pixels.shape[:2]
            )
            rows.extend(prediction_rows(output, path.name, spec))
        audit.extend({"file_name": path.name, **row} for row in changes)
        if split == "val":
            label_path = Path(img2label_paths([str(path)])[0])
            labels = [
                list(map(float, line.split()))
                for line in label_path.read_text().splitlines()
                if line.strip()
            ]
            if labels:
                labels = torch.tensor(labels, device=device)
                boxes = ops.xywh2xyxy(labels[:, 1:5]) * labels.new_tensor(
                    [width, height, width, height]
                )
                truth_rows.extend(
                    prediction_rows(
                        {
                            "bboxes": boxes,
                            "cls": labels[:, 0],
                            "conf": labels.new_ones(len(labels)),
                        },
                        path.name,
                        spec,
                    )
                )
    output_dir = output_root / f"predictions_{split}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in [("yolo", original_rows), ("ocr", revised_rows)]:
        table = pd.DataFrame(rows, columns=PRED_COLS)
        table.to_csv(output_dir / f"{name}_predictions.csv", index=False)
        if split == "test":
            # 사용자 지정: 외부 전용 클래스를 포함한 전체 예측을 그대로 제출한다.
            to_submission(table).to_csv(output_dir / f"submission_{name}.csv", index=False)
    pd.DataFrame(audit).to_csv(output_dir / "ocr_comparison.csv", index=False)
    if split == "val":
        pd.DataFrame(truth_rows, columns=PRED_COLS).to_csv(
            output_dir / "validation_truth.csv", index=False
        )
    (output_dir / "prediction_config.json").write_text(
        json.dumps(
            {
                "weights": str(weights.resolve()),
                "split": split,
                "image_count": len(images),
                "imgsz": imgsz,
                "conf": conf,
                "nms_iou": nms_iou,
                "max_det": max_det,
                "include_external_classes": True,
                "joint_loss_weight": spec["loss_weight"],
                "rerank": spec["rerank"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir

"""하나의 YOLO 특징맵에서 검출과 OCR CTC 손실을 함께 최적화한다."""

from copy import copy
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor, nn
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.torch_utils import unwrap_model

from models.evaluate import PRED_COLS
from models.ocr import decode_ctc, edit_distance, evaluate_ocr_predictions, rerank_with_ocr
from src.image_io import install_image_read_retry
from src.joint_ocr_data import JointYOLODataset, save_joint_spec
from src.joint_roi import eager_roi_align


def shared_feature(predictions: object) -> Tensor:
    """검출 헤드가 사용한 P3 특징맵을 detach 없이 반환한다."""
    raw = predictions[1] if isinstance(predictions, (tuple, list)) else predictions
    if not isinstance(raw, dict) or "feats" not in raw:
        raise RuntimeError(
            "공동 OCR은 feats를 반환하는 현재 Ultralytics YOLO11 검출 헤드가 필요합니다."
        )
    return raw["feats"][0]


class SharedOCRHead(nn.Module):
    """YOLO의 공유 특징에서 ROIAlign과 양방향 LSTM으로 각인을 읽는다."""

    def __init__(self, channels: int, hidden_size: int, classes: int, steps: int) -> None:
        super().__init__()
        self.steps = steps
        self.projection = nn.Sequential(nn.Conv2d(channels, 128, 1), nn.ReLU())
        self.sequence = nn.LSTM(128, hidden_size, num_layers=2, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, classes)

    def forward(self, feature: Tensor, rois: Tensor) -> Tensor:
        """배치 번호와 정규화 xyxy ROI에서 시간×박스×문자 logits를 구한다."""
        scale = feature.new_tensor(
            [1, feature.shape[3], feature.shape[2], feature.shape[3], feature.shape[2]]
        )
        with torch.autocast(device_type=feature.device.type, enabled=False):
            pooled = eager_roi_align(
                feature.float(),
                rois.float() * scale.float(),
                4,
                self.steps,
            )
            sequence = self.projection(pooled).mean(2).permute(2, 0, 1)
            sequence, _ = self.sequence(sequence)
            return self.classifier(sequence)


class JointDetectionModel(DetectionModel):
    """검출과 OCR 헤드가 같은 특징 추출기와 하나의 optimizer를 공유한다."""

    def __init__(
        self, cfg: dict | str, spec: dict, nc: int, ch: int = 3, verbose: bool = True
    ) -> None:
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        self.joint_spec = spec
        channels = self.model[-1].cv2[0][0].conv.in_channels
        self.ocr_head = SharedOCRHead(
            channels, spec["hidden_size"], len(spec["characters"]) + 1, spec["sequence_length"]
        )

    def ocr_targets(self, batch: dict) -> tuple[Tensor, list[str]]:
        """동일 검출 배치의 각인 정답이 있는 bbox만 OCR ROI로 고른다."""
        texts = batch.get("ocr_text", [])
        selected = [index for index, text in enumerate(texts) if text]
        if not selected:
            return batch["bboxes"].new_empty((0, 5)), []
        indices = torch.tensor(selected, device=batch["bboxes"].device)
        boxes = ops.xywh2xyxy(batch["bboxes"][indices])
        rois = torch.cat([batch["batch_idx"][indices, None].to(boxes), boxes], dim=1)
        return rois, [texts[index] for index in selected]

    def loss(self, batch: dict, preds: object = None) -> tuple[Tensor, dict]:
        """검출 손실 + 사용자 지정 가중치×CTC를 한 번에 역전파한다."""
        if preds is None:
            preds = self.forward(batch["img"])
        detection_loss, items = super().loss(batch, preds)
        feature = shared_feature(preds)
        rois, texts = self.ocr_targets(batch)
        # OCR 정답이 없는 배치도 공유 특징과 OCR 파라미터의 그래프를 유지한다.
        ctc = feature.sum() * 0 + sum(
            parameter.sum() * 0 for parameter in self.ocr_head.parameters()
        )
        if texts:
            logits = self.ocr_head(feature, rois)
            to_index = {
                character: index + 1
                for index, character in enumerate(self.joint_spec["characters"])
            }
            valid = [i for i, text in enumerate(texts) if set(text).issubset(to_index)]
            if valid:
                targets = torch.tensor(
                    [to_index[c] for i in valid for c in texts[i]], dtype=torch.long
                )
                lengths = torch.tensor([len(texts[i]) for i in valid], dtype=torch.long)
                probabilities = logits[:, valid].float().log_softmax(-1).cpu()
                steps = torch.full((len(valid),), logits.shape[0], dtype=torch.long)
                ctc = nn.functional.ctc_loss(
                    probabilities, targets, steps, lengths, blank=0, zero_infinity=False
                )
                ctc = ctc.to(feature.device)
        total = detection_loss.sum() + self.joint_spec["loss_weight"] * ctc * batch["img"].shape[0]
        return total, {**items, "ocr_loss": ctc.detach()}


def prediction_rows(prediction: dict, name: str, spec: dict) -> list[list]:
    """동일 좌표계의 검출 결과를 공통 평가 표로 바꾼다."""
    to_category = {row["yolo_idx"]: row["category_id"] for row in spec["class_map"]}
    boxes = prediction["bboxes"].detach().cpu().tolist()
    indices = prediction["cls"].int().cpu().tolist()
    scores = prediction["conf"].cpu().tolist()
    return [
        [name, index, to_category[index], x1, y1, x2 - x1, y2 - y1, score]
        for (x1, y1, x2, y2), index, score in zip(boxes, indices, scores, strict=True)
    ]


def revise_predictions(
    model: JointDetectionModel,
    feature: Tensor,
    predictions: list[dict],
    shape: tuple,
) -> tuple[list[dict], list[dict]]:
    """공유 특징으로 검출 ROI를 읽어 공동 모델의 최종 클래스를 보정한다."""
    spec = model.joint_spec
    settings = spec["rerank"]
    to_category = {row["yolo_idx"]: row["category_id"] for row in spec["class_map"]}
    to_index = {category: index for index, category in to_category.items()}
    dictionary = {int(category): texts for category, texts in spec["dictionary"].items()}
    revised, audit = [], []
    for image_index, prediction in enumerate(predictions):
        result = {key: value.clone() for key, value in prediction.items()}
        if len(prediction["cls"]):
            boxes = prediction["bboxes"] / feature.new_tensor(
                [shape[1], shape[0], shape[1], shape[0]]
            )
            rois = torch.cat([boxes.new_full((len(boxes), 1), image_index), boxes], dim=1)
            recognized = decode_ctc(model.ocr_head(feature, rois), spec["characters"])
            for index, (text, confidence) in enumerate(recognized):
                category = to_category[int(prediction["cls"][index])]
                category_new, score, reason = rerank_with_ocr(
                    category,
                    float(prediction["conf"][index]),
                    text,
                    confidence,
                    dictionary,
                    allowed_categories=set(to_index),
                    **settings,
                )
                result["cls"][index] = to_index[category_new]
                result["conf"][index] = score
                audit.append(
                    dict(
                        image_index=image_index,
                        box_index=index,
                        original_category=category,
                        category_id=category_new,
                        original_score=float(prediction["conf"][index]),
                        adjusted_score=score,
                        ocr_text=text,
                        ocr_confidence=confidence,
                        reason=reason,
                    )
                )
        revised.append(result)
    return revised, audit


class JointValidator(DetectionValidator):
    """매 에폭 같은 검증 배치에서 검출·OCR·융합 지표를 함께 기록한다."""

    def init_metrics(self, model: nn.Module) -> None:
        super().init_metrics(model)
        self.joint_model = model if isinstance(model, JointDetectionModel) else model.model
        if not isinstance(self.joint_model, JointDetectionModel):
            raise TypeError("공동 학습 체크포인트가 필요합니다.")
        # 검출기가 FP16으로 로드되어도 OCR/CTC는 FP32로 계산한다.
        self.joint_model.ocr_head.float()
        self.original_rows, self.revised_rows, self.truth_rows = [], [], []
        self.character_errors = self.character_count = self.exact_count = self.text_count = 0

    def postprocess(self, preds: object) -> list[dict]:
        self.joint_feature = shared_feature(preds)
        return super().postprocess(preds)

    def update_metrics(self, preds: list[dict], batch: dict) -> None:
        super().update_metrics(preds, batch)
        model = self.joint_model
        revised, _ = revise_predictions(model, self.joint_feature, preds, batch["img"].shape[2:])
        rois, texts = model.ocr_targets(batch)
        if texts:
            recognized = decode_ctc(
                model.ocr_head(self.joint_feature, rois), model.joint_spec["characters"]
            )
            for text, (recognized_text, _) in zip(texts, recognized, strict=True):
                self.character_errors += edit_distance(text, recognized_text)
                self.character_count += len(text)
                self.exact_count += int(text == recognized_text)
                self.text_count += 1
        for index, (original, final) in enumerate(zip(preds, revised, strict=True)):
            name = Path(batch["im_file"][index]).name
            self.original_rows.extend(prediction_rows(original, name, model.joint_spec))
            self.revised_rows.extend(prediction_rows(final, name, model.joint_spec))
            targets = self._prepare_batch(index, batch)
            targets["conf"] = targets["cls"].new_ones(len(targets["cls"]))
            self.truth_rows.extend(prediction_rows(targets, name, model.joint_spec))
        del self.joint_feature

    def get_stats(self) -> dict:
        stats = super().get_stats()
        truth = pd.DataFrame(self.truth_rows, columns=PRED_COLS)
        self.comparison = pd.DataFrame(
            {
                "YOLO": evaluate_ocr_predictions(
                    pd.DataFrame(self.original_rows, columns=PRED_COLS), truth
                ),
                "YOLO + OCR": evaluate_ocr_predictions(
                    pd.DataFrame(self.revised_rows, columns=PRED_COLS), truth
                ),
            }
        ).T
        stats["metrics/mAP75-95(B)"] = float(self.comparison.loc["YOLO", "map75_95"])
        stats.update(
            {
                f"joint/{key}": float(value)
                for key, value in self.comparison.loc["YOLO + OCR"].items()
            }
        )
        stats["ocr/cer"] = (
            self.character_errors / self.character_count if self.character_count else float("nan")
        )
        stats["ocr/exact_accuracy"] = (
            self.exact_count / self.text_count if self.text_count else float("nan")
        )
        stats["ocr/val_texts"] = self.text_count
        stats["fitness"] = stats["joint/map75_95"]
        return stats


class JointTrainer(DetectionTrainer):
    """기존 YOLO 학습 흐름에 공동 모델·동일 배치·통합 검증만 추가한다."""

    def __init__(self, *args: object, joint_spec: dict, **kwargs: object) -> None:
        self.joint_spec = joint_spec
        install_image_read_retry()
        super().__init__(*args, **kwargs)
        if self.world_size > 1:
            raise ValueError("공동 OCR 검증은 현재 단일 장치에서 실행하세요.")
        save_joint_spec(joint_spec, self.save_dir)

    def train(self) -> None:
        """공동 학습 범위에서 내부 자동 컴파일까지 끄고 종료 후 원래 상태로 복원한다."""
        self.args.compile = False
        with torch.compiler.set_stance("force_eager"):
            super().train()

    def build_dataset(
        self, img_path: str, mode: str = "train", batch: int | None = None
    ) -> JointYOLODataset:
        dataset = super().build_dataset(img_path, mode, batch)
        dataset.__class__ = JointYOLODataset
        dataset.bind_joint_texts(self.joint_spec)
        return dataset

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: nn.Module | None = None,
        verbose: bool = True,
    ) -> JointDetectionModel:
        model = self.set_model_names_for_load(
            JointDetectionModel(
                cfg,
                self.joint_spec,
                self.data["nc"],
                self.data["channels"],
                verbose,
            )
        )
        if weights is not None:
            model.load(weights)
        return model

    def get_validator(self) -> JointValidator:
        return JointValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def validate(self) -> tuple[dict | None, float | None]:
        """기존 노트북의 저장·중단 정책을 유지하면서 융합 mAP를 fitness로 사용한다."""
        # 다중 상속에서 기존 NotebookTrainer의 검출 단독 fitness 덮어쓰기를 건너뛴다.
        metrics, fitness = DetectionTrainer.validate(self)
        if metrics is not None:
            metrics["speed/inference_ms"] = float(self.validator.speed["inference"])
            LOGGER.info(f"공동 epoch {self.epoch + 1}: 융합 mAP75-95={fitness:.5f}")
        return metrics, fitness

    def build_optimizer(
        self,
        model: nn.Module,
        name: str = "auto",
        lr: float = 0.001,
        momentum: float = 0.9,
        decay: float = 1e-5,
        iterations: int = 100000,
    ) -> torch.optim.Optimizer:
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
        trained = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        required = {
            id(parameter)
            for parameter in unwrap_model(model).parameters()
            if parameter.requires_grad
        }
        if not required.issubset(trained):
            raise RuntimeError("공동 optimizer에서 누락되거나 불필요한 파라미터가 있습니다.")
        return optimizer

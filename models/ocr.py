"""YOLO와 별도로 학습하는 CRNN·CTC 각인 인식과 보수적인 클래스 재분류."""

import json
import random
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.config import get_device
from src.ocr_data import image_path, ocr_tensor, pill_crop, prepare_crop_cache, true_values


class PillCRNN(nn.Module):
    """알약 crop에서 문자 시퀀스의 시간별 logits를 계산한다."""

    def __init__(self, classes: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.sequence = nn.LSTM(128, hidden_size, num_layers=2, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, classes)

    def forward(self, images: Tensor) -> Tensor:
        """시간 × 배치 × 문자 클래스 형태의 logits를 반환한다."""
        features = self.features(images).squeeze(2).permute(2, 0, 1)
        sequence, _ = self.sequence(features)
        return self.classifier(sequence)


class PillTextDataset(Dataset):
    """bbox crop과 선택한 출처의 각인 문자열을 OCR 학습 입력으로 읽는다."""

    def __init__(
        self,
        rows: pd.DataFrame,
        characters: str,
        height: int,
        width: int,
        encode_targets: bool = True,
        cache_path: Path | None = None,
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.to_index = {character: index + 1 for index, character in enumerate(characters)}
        self.height, self.width = height, width
        self.encode_targets = encode_targets
        self.cache_path = cache_path
        self.cache = None
        for text in self.rows["text"]:
            if encode_targets and set(text) - set(characters):
                raise ValueError(f"학습 문자 사전에 없는 문자가 있습니다: {text!r}")
            required = len(text) + sum(a == b for a, b in zip(text, text[1:], strict=False))
            if encode_targets and required > width // 4:
                raise ValueError(
                    f"CTC 출력 길이보다 긴 각인입니다. 입력 너비를 확인하세요: {text!r}"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        row = self.rows.iloc[index]
        if self.cache_path is not None:
            if self.cache is None:
                self.cache = np.load(self.cache_path, mmap_mode="r", allow_pickle=False)
            pixels = np.array(self.cache[int(row["_crop_index"])], dtype=np.float32)
            tensor = torch.from_numpy(pixels)[None] / 127.5 - 1.0
        else:
            with Image.open(image_path(str(row["path"]))) as image:
                crop = pill_crop(image, row[["x", "y", "w", "h"]].to_list())
                tensor = ocr_tensor(crop, self.height, self.width)
        target = torch.tensor(
            [self.to_index[c] for c in row["text"]] if self.encode_targets else [], dtype=torch.long
        )
        return tensor, target, row["text"]


def collate_text(batch: list[tuple[Tensor, Tensor, str]]) -> tuple:
    """가변 길이 문자열의 CTC 정답을 이어 붙인다."""
    images, targets, texts = zip(*batch, strict=True)
    return (
        torch.stack(images),
        torch.cat(targets),
        torch.tensor([len(target) for target in targets], dtype=torch.long),
        list(texts),
    )


def decode_ctc(logits: Tensor, characters: str) -> list[tuple[str, float]]:
    """CTC 반복 문자·blank를 제거하고 선택된 문자 확률 평균을 반환한다."""
    probabilities, indices = logits.detach().softmax(-1).max(-1)
    results = []
    for tokens, scores in zip(indices.T.tolist(), probabilities.T.tolist(), strict=True):
        text, confidence, previous = [], [], -1
        for token, score in zip(tokens, scores, strict=True):
            if token and token != previous:
                text.append(characters[token - 1])
                confidence.append(score)
            previous = token
        results.append(("".join(text), float(np.mean(confidence)) if confidence else 0.0))
    return results


def edit_distance(first: str, second: str) -> int:
    """문자 오류율 계산에 필요한 편집 거리를 구한다."""
    previous = list(range(len(second) + 1))
    for index, character in enumerate(first, 1):
        current = [index]
        for position, other in enumerate(second, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[position] + 1,
                    previous[position - 1] + (character != other),
                )
            )
        previous = current
    return previous[-1]


def _label_rows(rows: pd.DataFrame, label_mode: str) -> pd.DataFrame:
    """명시적으로 허용한 각인 출처만 선택하고 사람 검수 여부는 바꾸지 않는다."""
    if label_mode not in {"reviewed", "metadata"}:
        raise ValueError("OCR label_mode는 reviewed 또는 metadata여야 합니다.")
    if rows.empty or not {"decision", "reviewed", "text", "split"}.issubset(rows.columns):
        raise ValueError("OCR 학습에는 검수된 train/val 표가 필요합니다.")
    valid = rows["decision"].eq("승인") & true_values(rows["reviewed"])
    if label_mode == "metadata":
        if not {"label_source", "suggested_text"}.issubset(rows):
            raise ValueError("JSON 메타데이터 출처와 원래 각인 후보가 필요합니다.")
        suggested = rows["suggested_text"].fillna("").astype(str).str.strip()
        weak = (
            rows["label_source"].eq("json_metadata")
            & ~true_values(rows["reviewed"])
            & rows["decision"].eq("")
            & suggested.ne("")
            & ~suggested.str.contains("마크|분할선|로고|없음|무각인|^[\\W_]+$", regex=True)
            & rows["text"].eq(suggested)
        )
        valid |= weak
    if "active" in rows:
        valid &= true_values(rows["active"])
    rows = rows[valid].copy()
    rows["text"] = rows["text"].fillna("").astype(str).str.strip()
    rows = rows[rows["text"].ne("")].copy()
    return rows


def _approved_rows(
    rows: pd.DataFrame, label_mode: str = "reviewed", *, all_train: bool = False
) -> pd.DataFrame:
    """허용한 정답 출처와 train/val 조합 분리를 학습 시작 전에 검증한다."""
    rows = _label_rows(rows, label_mode)
    if all_train:
        if rows.empty or not set(rows["split"]).issubset({"train", "val"}):
            raise ValueError("전체 OCR 학습에는 train/val 출처의 각인 정답이 필요합니다.")
        rows["split"] = "train"
        return rows
    if not set(rows["split"]).issubset({"train", "val"}):
        raise ValueError("OCR 학습 분할은 train/val만 허용합니다.")
    train = rows[rows["split"].eq("train")]
    val = rows[rows["split"].eq("val")]
    if train.empty or val.empty:
        raise ValueError("OCR train과 val 양쪽에 허용된 출처의 각인 정답이 필요합니다.")
    train_combos = set(train["file_name"].str.split("_").str[0])
    if train_combos & set(val["file_name"].str.split("_").str[0]):
        raise ValueError("OCR train/val 이미지 조합이 겹칩니다.")
    return rows


def train_ocr(
    rows: pd.DataFrame,
    output_dir: Path,
    *,
    epochs: int = 30,
    height: int = 32,
    width: int = 160,
    batch: int = 32,
    lr: float = 0.001,
    seed: int = 42,
    workers: int = 0,
    hidden_size: int = 128,
    label_mode: str = "reviewed",
    cache_crops: bool = True,
    optimizer: str = "AdamW",
    all_train: bool = False,
) -> Path:
    """독립 OCR을 학습해 검증 CER 최저 모델 또는 전체 학습의 마지막 모델을 반환한다."""
    rows = _approved_rows(rows, label_mode, all_train=all_train)
    rows["_crop_index"] = range(len(rows))
    if epochs < 1 or batch < 1 or lr <= 0 or workers < 0 or height < 4 or width < 4:
        raise ValueError("OCR 학습 인자 범위를 확인하세요.")
    best_path = output_dir / ("ocr_last.pt" if all_train else "ocr_best.pt")
    if best_path.exists():
        raise FileExistsError(f"기존 OCR 모델을 덮어쓰지 않습니다: {best_path}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train = rows[rows["split"].eq("train")]
    val = rows[rows["split"].eq("val")]
    characters = "".join(sorted(set("".join(train["text"]))))
    cache_path = output_dir / "crop_cache.npy" if cache_crops else None
    train_dataset = PillTextDataset(train, characters, height, width, cache_path=cache_path)
    # 검증에만 등장하는 문자는 정답 문자열에 남겨 CER 오류로 계산한다.
    val_dataset = PillTextDataset(
        val, characters, height, width, encode_targets=False, cache_path=cache_path
    )
    if cache_path is not None:
        prepare_crop_cache(rows, cache_path, height, width)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_text,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch, num_workers=workers, collate_fn=collate_text
    )
    device = get_device()
    model = PillCRNN(len(characters) + 1, hidden_size).to(device)
    optimizer_name = optimizer
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == "MuSGD":
        from ultralytics.optim import MuSGD

        # 행렬·합성곱에만 Muon을 적용하고 편향은 SGD로 갱신한다.
        optimizer = MuSGD(
            [
                {"params": [p for p in model.parameters() if p.ndim >= 2], "use_muon": True},
                {"params": [p for p in model.parameters() if p.ndim < 2], "use_muon": False},
            ],
            lr=lr,
        )
    else:
        raise ValueError("OCR optimizer는 AdamW 또는 MuSGD여야 합니다.")
    criterion = nn.CTCLoss(blank=0, zero_infinity=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(
        epochs=epochs,
        height=height,
        width=width,
        batch=batch,
        lr=lr,
        seed=seed,
        workers=workers,
        hidden_size=hidden_size,
        optimizer=optimizer_name,
        all_train=all_train,
        checkpoint_selection="last_epoch" if all_train else "lowest_val_cer",
        optimizer_defaults=optimizer.defaults,
        device=device,
        label_mode=label_mode,
        cache_crops=cache_crops,
        external_val_fraction=rows.attrs.get("external_val_fraction"),
        split_seed=rows.attrs.get("split_seed"),
        shared_split_path=rows.attrs.get("shared_split_path"),
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows.to_csv(output_dir / "label_split.csv", index=False)
    history, best_cer = [], float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for images, targets, lengths, _ in tqdm(
            train_loader, desc=f"OCR epoch {epoch}/{epochs}", mininterval=5, leave=False
        ):
            logits = model(images.to(device))
            # CTC 손실은 장치별 지원 차이를 피하기 위해 CPU에서 계산한다.
            probabilities = logits.float().log_softmax(-1).to("cpu")
            steps = torch.full((images.shape[0],), logits.shape[0], dtype=torch.long)
            loss = criterion(probabilities, targets, steps, lengths)
            if not torch.isfinite(loss):
                raise ValueError("CTC loss가 유한하지 않습니다. 각인 길이와 입력을 확인하세요.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(images)
        model.eval()
        errors, total_chars, exact = 0, 0, 0
        with torch.no_grad():
            for images, _, _, texts in val_loader:
                predictions = decode_ctc(model(images.to(device)), characters)
                for (predicted, _), truth in zip(predictions, texts, strict=True):
                    errors += edit_distance(truth, predicted)
                    total_chars += len(truth)
                    exact += predicted == truth
        cer = None if all_train else errors / max(total_chars, 1)
        record = dict(
            epoch=epoch,
            train_loss=loss_sum / len(train),
        )
        if not all_train:
            record.update(val_cer=cer, val_exact_accuracy=exact / len(val))
        history.append(record)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        print(record, flush=True)
        if all_train or cer < best_cer:
            if not all_train:
                best_cer = cer
            checkpoint = dict(
                state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                characters=characters,
                config=config,
                epoch=epoch,
                val_cer=cer,
            )
            temporary = best_path.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(best_path)
    return best_path


class OCRRecognizer:
    """독립 OCR 체크포인트로 알약 crop의 문자열과 확률 요약을 반환한다."""

    def __init__(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.characters = checkpoint["characters"]
        self.config = checkpoint["config"]
        self.device = get_device()
        self.model = PillCRNN(len(self.characters) + 1, self.config["hidden_size"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def read(self, image: Image.Image, bbox: list[float]) -> tuple[str, float]:
        """YOLO가 예측한 원본 픽셀 bbox에서 각인을 읽는다."""
        crop = pill_crop(image, bbox)
        tensor = ocr_tensor(crop, self.config["height"], self.config["width"])
        return decode_ctc(self.model(tensor[None].to(self.device)), self.characters)[0]


def build_imprint_dictionary(
    rows: pd.DataFrame, *, label_mode: str = "reviewed"
) -> dict[int, list[str]]:
    """명시한 출처의 OCR train 문자열만 각인 사전에 넣어 검증 정답 누수를 막는다."""
    selected = _label_rows(rows, label_mode)
    selected = selected[selected["split"].eq("train")].copy()
    selected["text"] = selected["text"].fillna("").astype(str).str.strip()
    return {
        int(category): sorted({text for text in group["text"] if text})
        for category, group in selected.groupby("category_id")
    }


def rerank_with_ocr(
    category_id: int,
    score: float,
    text: str,
    ocr_confidence: float,
    dictionary: dict[int, list[str]],
    *,
    min_confidence: float,
    min_similarity: float,
    min_margin: float,
    weight: float,
    allowed_categories: set[int] | None = None,
) -> tuple[int, float, str]:
    """명확한 각인 일치가 있을 때만 클래스를 수정하고 애매하면 YOLO 결과를 유지한다."""
    if not all(
        0 <= value <= 1
        for value in [score, ocr_confidence, min_confidence, min_similarity, min_margin, weight]
    ):
        raise ValueError("확률·유사도·가중치는 0~1이어야 합니다.")
    if weight == 0 or not text.strip() or ocr_confidence < min_confidence:
        return category_id, score, "OCR 근거 부족: YOLO 유지"

    def normalized(value: str) -> str:
        return "".join(value.upper().split())

    recognized = normalized(text)
    if not recognized:
        return category_id, score, "OCR 문자열 없음: YOLO 유지"
    candidates = []
    for category, variants in dictionary.items():
        if allowed_categories is not None and category not in allowed_categories:
            continue
        similarities = [
            SequenceMatcher(None, recognized, normalized(v)).ratio()
            for v in variants
            if normalized(v)
        ]
        if similarities:
            candidates.append((max(similarities), category))
    candidates.sort(reverse=True)
    if not candidates:
        return category_id, score, "각인 사전 없음: YOLO 유지"
    similarity, category = candidates[0]
    second = candidates[1][0] if len(candidates) > 1 else 0.0
    if similarity < min_similarity or similarity <= second or similarity - second < min_margin:
        return category_id, score, "각인 일치 모호: YOLO 유지"
    if category == category_id:
        return category_id, score, "YOLO·OCR 클래스 일치"
    # 새 점수는 보정된 확률이 아니라 검증이 필요한 휴리스틱 점수다.
    adjusted = score * ((1 - weight) + weight * ocr_confidence * similarity)
    return category, adjusted, "각인 기반 클래스 수정: 검증 필요"


def predict_with_ocr(
    yolo_weights: Path,
    ocr_weights: Path,
    images: list[Path],
    class_map: pd.DataFrame,
    dictionary: dict[int, list[str]],
    output_dir: Path,
    *,
    imgsz: int,
    conf: float,
    nms_iou: float,
    max_det: int,
    min_confidence: float,
    min_similarity: float,
    min_margin: float,
    weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """같은 YOLO 박스에 OCR을 적용하고 원래 예측·재분류 예측·비교표를 저장한다."""
    from ultralytics import YOLO

    from models.evaluate import PRED_COLS, to_submission

    if not images or not all(path.is_file() for path in images):
        raise FileNotFoundError("OCR 재분류에 사용할 이미지 경로를 확인하세요.")
    if class_map["yolo_idx"].duplicated().any() or class_map["category_id"].duplicated().any():
        raise ValueError("YOLO 클래스 매핑에 중복이 있습니다.")
    model = YOLO(yolo_weights)
    expected = dict(zip(class_map["yolo_idx"], class_map["dl_name"], strict=True))
    if model.names != expected:
        raise ValueError("YOLO 체크포인트와 클래스 매핑이 다릅니다.")
    to_category = dict(zip(class_map["yolo_idx"], class_map["category_id"], strict=True))
    to_index = {int(category): int(index) for index, category in to_category.items()}
    recognizer = OCRRecognizer(ocr_weights)
    original_rows, adjusted_rows, audit = [], [], []
    # 경로 목록을 넘기면 전체 원본이 메모리에 올라가므로 한 장씩 읽는다.
    results = (
        result
        for path in tqdm(images, desc="YOLO + OCR 예측", mininterval=5)
        for result in model.predict(
            source=str(path),
            stream=True,
            imgsz=imgsz,
            conf=conf,
            iou=nms_iou,
            max_det=max_det,
            device=get_device(),
            verbose=False,
        )
    )
    for result in results:
        image = Image.fromarray(result.orig_img[..., ::-1])
        for box, index, score in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.cls.int().cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            strict=True,
        ):
            x1, y1, x2, y2 = box
            bbox = [x1, y1, x2 - x1, y2 - y1]
            text, confidence = recognizer.read(image, bbox)
            category = int(to_category[index])
            revised, new_score, reason = rerank_with_ocr(
                category,
                score,
                text,
                confidence,
                dictionary,
                min_confidence=min_confidence,
                min_similarity=min_similarity,
                min_margin=min_margin,
                weight=weight,
                allowed_categories=set(to_index),
            )
            name = Path(result.path).name
            original_rows.append([name, index, category, *bbox, score])
            adjusted_rows.append([name, to_index[revised], revised, *bbox, new_score])
            audit.append(
                dict(
                    file_name=name,
                    original_category=category,
                    category_id=revised,
                    original_score=score,
                    adjusted_score=new_score,
                    ocr_text=text,
                    ocr_confidence=confidence,
                    reason=reason,
                )
            )
    if not original_rows:
        raise ValueError("YOLO 예측 박스가 없어 OCR 재분류 결과를 저장하지 않았습니다.")
    original = pd.DataFrame(original_rows, columns=PRED_COLS)
    adjusted = pd.DataFrame(adjusted_rows, columns=PRED_COLS)
    output_dir.mkdir(parents=True, exist_ok=True)
    original.to_csv(output_dir / "yolo_predictions.csv", index=False)
    adjusted.to_csv(output_dir / "ocr_predictions.csv", index=False)
    pd.DataFrame(audit).to_csv(output_dir / "ocr_comparison.csv", index=False)
    # 대회 test의 숫자 파일명일 때만 Kaggle 형식으로 변환한다.
    if all(Path(name).stem.isdecimal() for name in original["file_name"].unique()):
        to_submission(original).to_csv(output_dir / "submission_yolo.csv", index=False)
        to_submission(adjusted).to_csv(output_dir / "submission_ocr.csv", index=False)
    return original, adjusted


def evaluate_ocr_predictions(predictions: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float]:
    """동일 validation 이미지의 저장 예측을 IoU 0.50~0.95 AP로 평가한다."""
    from ultralytics.utils.metrics import ap_per_class

    if truth.empty:
        raise ValueError("OCR 재분류 비교에 validation 정답이 필요합니다.")
    if predictions.empty:
        return dict(precision=0.0, recall=0.0, map50=0.0, map50_95=0.0, map75_95=0.0)
    if set(predictions["file_name"]) - set(truth["file_name"]):
        raise ValueError("validation 정답에 없는 이미지의 예측이 섞였습니다.")
    thresholds = np.linspace(0.5, 0.95, 10)
    correct, confidences, categories = [], [], []
    for name, targets in truth.groupby("file_name"):
        detected = predictions[predictions["file_name"].eq(name)].sort_values(
            "conf", ascending=False
        )
        target_boxes = targets[["x", "y", "w", "h"]].to_numpy(dtype=float, copy=True)
        pred_boxes = detected[["x", "y", "w", "h"]].to_numpy(dtype=float, copy=True)
        target_boxes[:, 2:] += target_boxes[:, :2]
        pred_boxes[:, 2:] += pred_boxes[:, :2]
        intersections = np.maximum(
            0,
            np.minimum(target_boxes[:, None, 2:], pred_boxes[None, :, 2:])
            - np.maximum(target_boxes[:, None, :2], pred_boxes[None, :, :2]),
        ).prod(axis=-1)
        area_true = (target_boxes[:, 2:] - target_boxes[:, :2]).prod(axis=1)
        area_pred = (pred_boxes[:, 2:] - pred_boxes[:, :2]).prod(axis=1)
        overlaps = intersections / np.maximum(
            area_true[:, None] + area_pred[None, :] - intersections, 1e-9
        )
        overlaps *= (
            targets["category_id"].to_numpy()[:, None]
            == detected["category_id"].to_numpy()[None, :]
        )
        matches = np.zeros((len(detected), 10), dtype=bool)
        claimed = np.zeros((len(targets), 10), dtype=bool)
        for index in range(len(detected)):
            available = np.where(claimed, 0.0, overlaps[:, index, None])
            best = available.argmax(axis=0)
            matches[index] = available[best, np.arange(10)] >= thresholds
            claimed[best, np.arange(10)] |= matches[index]
        correct.append(matches)
        confidences.extend(detected["conf"].to_list())
        categories.extend(detected["category_id"].to_list())
    metrics = ap_per_class(
        np.concatenate(correct),
        np.asarray(confidences),
        np.asarray(categories),
        truth["category_id"].to_numpy(),
    )
    precision, recall, ap = metrics[2], metrics[3], metrics[5]
    return {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "map50": float(ap[:, 0].mean()),
        "map50_95": float(ap.mean()),
        "map75_95": float(ap[:, 5:].mean()),
    }

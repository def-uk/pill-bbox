"""공동 학습에서 기존 YOLO 입력과 각인 정답의 이미지·분할을 일치시킨다."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import img2label_paths

from models.ocr import _label_rows
from src.config import PROJECT_ROOT
from src.image_io import install_image_read_retry
from src.ocr_data import read_ocr_review_table, true_values


def image_lists(data_path: Path) -> dict[str, list[Path]]:
    """파이프라인이 만든 train/val 목록만 읽고 이미지·조합 누수를 검사한다."""
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(data.get("path", PROJECT_ROOT))
    inventory = {}
    for split in ["train", "val"]:
        listing = Path(data[split])
        if not listing.is_absolute():
            listing = root / listing
        paths = [Path(line.strip()) for line in listing.read_text().splitlines() if line.strip()]
        paths = [path if path.is_absolute() else listing.parent / path for path in paths]
        if len({path.name for path in paths}) != len(paths):
            raise ValueError(f"{split}: 이미지 파일명 중복이 있습니다.")
        inventory[split] = paths
    for key in [lambda path: path.name, lambda path: path.stem.split("_")[0]]:
        if {key(p) for p in inventory["train"]} & {key(p) for p in inventory["val"]}:
            raise ValueError("공동 학습 train/val 이미지 또는 조합이 겹칩니다.")
    return inventory


def prepare_joint_spec(
    data_path: Path,
    class_map: pd.DataFrame,
    review_path: Path,
    *,
    label_mode: str,
    loss_weight: float,
    hidden_size: int,
    sequence_length: int,
) -> dict:
    """동일 YOLO 목록에서 학습 문자 사전과 OCR 정답·출처·커버리지를 만든다."""
    if not np.isfinite(loss_weight) or loss_weight <= 0:
        raise ValueError("공동 OCR 손실 가중치는 양수여야 합니다.")
    inventory = image_lists(data_path)
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    expected_names = dict(zip(class_map["yolo_idx"], class_map["dl_name"], strict=True))
    if data["names"] != expected_names or class_map["category_id"].duplicated().any():
        raise ValueError("공동 클래스 표와 실제 YOLO data.yaml의 클래스가 다릅니다.")
    mapping = {path.name: split for split, paths in inventory.items() for path in paths}
    review = read_ocr_review_table(review_path)
    review = review.loc[review["file_name"].isin(mapping)].copy()
    review["split"] = review["file_name"].map(mapping)
    reviewed = true_values(review["reviewed"])
    review["label_source"] = np.where(reviewed, "human_reviewed", "json_metadata")
    if label_mode == "metadata":
        review.loc[~reviewed, "text"] = review.loc[~reviewed, "suggested_text"]
    rows = _label_rows(review, label_mode)
    keys = ["file_name", "category_id"]
    if rows.duplicated(keys).any():
        raise ValueError("같은 이미지·클래스의 OCR 정답이 중복됩니다. 원본 bbox를 확인하세요.")
    if not set(rows["category_id"]).issubset(set(class_map["category_id"])):
        raise ValueError("OCR 정답 클래스가 YOLO 클래스 표에 없습니다.")
    train = rows.loc[rows["split"].eq("train")]
    if train.empty or rows.loc[rows["split"].eq("val")].empty:
        raise ValueError("같은 YOLO train/val 양쪽에 OCR 정답이 필요합니다.")
    characters = "".join(sorted(set("".join(train["text"]))))
    for text in train["text"]:
        length = len(text) + sum(a == b for a, b in zip(text, text[1:], strict=False))
        if length > sequence_length:
            raise ValueError(f"OCR 출력 길이보다 긴 학습 각인입니다: {text!r}")
    dictionary = {
        int(category): sorted(set(group["text"]))
        for category, group in train.groupby("category_id")
    }
    return {
        "data_path": str(data_path.resolve()),
        "class_map": class_map.sort_values("yolo_idx").to_dict("records"),
        "characters": characters,
        "loss_weight": loss_weight,
        "hidden_size": hidden_size,
        "sequence_length": sequence_length,
        "label_mode": label_mode,
        "dictionary": dictionary,
        "rows": rows.to_dict("records"),
        "inventory": {split: [str(path) for path in paths] for split, paths in inventory.items()},
    }


def save_joint_spec(spec: dict, output_dir: Path) -> pd.DataFrame:
    """공통 분할과 OCR 정답 출처·클래스별 지원 범위를 실행 폴더에 저장한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "joint_spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = pd.DataFrame(spec["rows"])
    rows.to_csv(output_dir / "ocr_label_split.csv", index=False)
    inventory = pd.DataFrame(
        [
            {"file_name": Path(path).name, "path": path, "split": split}
            for split, paths in spec["inventory"].items()
            for path in paths
        ]
    )
    inventory.to_csv(output_dir / "shared_image_split.csv", index=False)
    coverage = pd.DataFrame(spec["class_map"])
    for split in ["train", "val"]:
        counts = rows.loc[rows["split"].eq(split), "category_id"].value_counts()
        coverage[f"ocr_{split}_bbox"] = coverage["category_id"].map(counts).fillna(0).astype(int)
        detection_counts = {}
        for path in img2label_paths(spec["inventory"][split]):
            for line in Path(path).read_text().splitlines():
                if line.strip():
                    index = int(line.split()[0])
                    detection_counts[index] = detection_counts.get(index, 0) + 1
        coverage[f"yolo_{split}_bbox"] = (
            coverage["yolo_idx"].map(detection_counts).fillna(0).astype(int)
        )
    coverage.to_csv(output_dir / "ocr_class_coverage.csv", index=False)
    return coverage


class JointYOLODataset(YOLODataset):
    """같은 증강을 거친 검출 bbox와 각인 정답을 인스턴스별로 연결한다."""

    def load_image(
        self,
        i: int,
        rect_mode: bool = True,
        resize_short: bool = False,
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """각 작업자에서도 일시적인 이미지 읽기 실패를 재시도한다."""
        install_image_read_retry()
        return super().load_image(i, rect_mode=rect_mode, resize_short=resize_short)

    def bind_joint_texts(self, spec: dict) -> None:
        """실제 YOLO 클래스는 보존하고 증강용 임시 식별자에 각인을 연결한다."""
        to_category = {row["yolo_idx"]: row["category_id"] for row in spec["class_map"]}
        texts = {(row["file_name"], row["category_id"]): row["text"] for row in spec["rows"]}
        self.joint_instance_ids = []
        self.joint_classes = []
        self.joint_texts = []
        for label in self.labels:
            ids = []
            for cls in label["cls"].reshape(-1).astype(int):
                ids.append(len(self.joint_classes))
                self.joint_classes.append(int(cls))
                self.joint_texts.append(
                    texts.get((Path(label["im_file"]).name, to_category[cls]), "")
                )
            self.joint_instance_ids.append(np.array(ids, dtype=np.float32).reshape(-1, 1))

    def get_image_and_label(self, index: int) -> dict:
        """Mosaic·회전·잘림에서도 bbox와 각인 연결이 함께 이동하도록 한다."""
        label = super().get_image_and_label(index)
        label["cls"] = self.joint_instance_ids[index].copy()
        return label

    def __getitem__(self, index: int) -> dict:
        """증강 완료 후 실제 클래스를 복구하고 같은 순서의 OCR 정답을 붙인다."""
        result = super().__getitem__(index)
        ids = result["cls"].reshape(-1).long().tolist()
        result["cls"] = torch.tensor(
            [self.joint_classes[i] for i in ids], dtype=torch.float32
        ).reshape(-1, 1)
        result["ocr_text"] = [self.joint_texts[i] for i in ids]
        return result

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """같은 이미지 배치에서 bbox와 가변 길이 문자열 순서를 일치시킨다."""
        texts = [sample.pop("ocr_text") for sample in batch]
        result = YOLODataset.collate_fn(batch)
        result["ocr_text"] = [text for group in texts for text in group]
        return result

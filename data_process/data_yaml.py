from pathlib import Path

import pandas as pd
import yaml

from src.config import PROJECT_ROOT

YOLO_ROOT = PROJECT_ROOT / "data" / "processed" / "yolo"
CLASS_MAP_COLS = ["yolo_idx", "category_id", "dl_name"]


def yolo_dir(label_version: str) -> Path:
    """라벨 버전의 YOLO 데이터 폴더(images/, labels/, class_map.csv, splits/)."""
    return YOLO_ROOT / label_version


def data_yaml_path(label_version: str, split_name: str, fold: int) -> Path:
    """학습에 넘길 fold별 data.yaml 경로."""
    return yolo_dir(label_version) / "splits" / split_name / f"fold_{fold}" / "data.yaml"


def build_class_map(labels: pd.DataFrame) -> pd.DataFrame:
    """category_id 오름차순으로 0부터 yolo_idx를 붙인 클래스 표를 만든다."""
    first = labels.sort_values("category_id").drop_duplicates("category_id")
    class_map = first[["category_id", "dl_name"]].reset_index(drop=True)
    class_map.insert(0, "yolo_idx", range(len(class_map)))
    return class_map[CLASS_MAP_COLS]


def load_class_map(label_version: str) -> pd.DataFrame:
    """저장된 class_map.csv를 읽는다. 예측의 yolo_idx를 category_id로 되돌릴 때 쓴다."""
    return pd.read_csv(yolo_dir(label_version) / "class_map.csv", encoding="utf-8")


def class_names(class_map: pd.DataFrame, use_code: bool = False) -> dict[int, str]:
    """data.yaml의 names. 기본은 한글 약 이름, use_code면 category_id 문자열이다.

    한글 폰트가 없는 환경(#50)에서 혼동 행렬 글자가 깨지면 use_code=True로 만든다.
    """
    col = "category_id" if use_code else "dl_name"
    return {
        int(i): str(name) for i, name in zip(class_map["yolo_idx"], class_map[col], strict=True)
    }


def write_fold_files(
    fold_rows: pd.DataFrame,
    images_dir: Path,
    out_dir: Path,
    names: dict[int, str],
    header: dict[str, object],
) -> Path:
    """fold 표(fold_k.csv 내용)로 train.txt·val.txt·data.yaml을 쓰고 data.yaml 경로를 반환한다.

    txt에는 images_dir 아래 이미지의 절대 경로를 적는다. ultralytics는 경로의 images/를
    labels/로 바꿔 라벨을 찾는다. header는 data.yaml 맨 위에 주석으로 남긴다(라벨 버전·분할·fold).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {"path": str(images_dir.parent.parent)}
    for split in ["train", "val"]:
        files = sorted(fold_rows.loc[fold_rows["split"] == split, "file_name"])
        txt_path = out_dir / f"{split}.txt"
        txt_path.write_text("".join(f"{images_dir / f}\n" for f in files), encoding="utf-8")
        data[split] = str(txt_path)
    data.update(nc=len(names), names=names)
    comments = "".join(f"# {key}: {value}\n" for key, value in header.items())
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        comments + yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return yaml_path

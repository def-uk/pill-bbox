"""노트북 공통 데이터 불러오기. 노트북 첫 셀에서 아래처럼 쓴다.

    from nb_data import ensure_pipeline, load_fold_labels, load_labels

노트북은 notebooks/에서 실행되므로 이 파일은 바로 import된다. import할 때 프로젝트 루트를
sys.path에 넣으므로 그 뒤로는 data_process, models, src도 그대로 import할 수 있다.
데이터는 data_process.data_main이 만든 산출물만 읽고, 노트북에서 따로 만들지 않는다.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from data_process.data_yaml import data_yaml_path, load_class_map, yolo_dir  # noqa: E402
from data_process.split import N_FOLDS  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

__all__ = [
    "PROCESSED_DIR",
    "PROJECT_ROOT",
    "data_yaml_path",
    "default_splits",
    "ensure_pipeline",
    "image_path",
    "load_class_map",
    "load_fold_labels",
    "load_folds",
    "load_labels",
]


def default_splits(label_version: str) -> list[str]:
    """data_main이 --split 없이 만드는 기본 분할 이름(combo, setup30fix)."""
    return [f"combo_{label_version}_s42", f"setup30fix_{label_version}_s42"]


def _labels_path(label_version: str) -> Path:
    return PROCESSED_DIR / "labels" / f"labels_{label_version}.csv"


def ensure_pipeline(
    label_version: str = "v1", split_names: list[str] | None = None, rerun: bool = False
) -> list[Path]:
    """라벨·fold·data.yaml이 없으면(rerun이면 항상) data_main을 실행하고 data.yaml 경로를 반환한다.

    data_main은 import하면 matplotlib을 Agg로 바꿔 노트북 그림이 안 보이므로 별도 프로세스로 돌린다.
    """
    split_names = split_names or default_splits(label_version)
    yaml_paths = [
        data_yaml_path(label_version, name, fold) for name in split_names for fold in range(N_FOLDS)
    ]
    expected = [_labels_path(label_version), *yaml_paths]
    if rerun or not all(p.exists() for p in expected):
        cmd = [sys.executable, "-m", "data_process.data_main", "--label", label_version]
        subprocess.run([*cmd, "--split", *split_names], cwd=PROJECT_ROOT, check=True)
    return yaml_paths


def load_labels(label_version: str = "v1") -> pd.DataFrame:
    """labels_<버전>.csv를 읽는다. bbox 한 개가 한 행이다."""
    path = _labels_path(label_version)
    if not path.exists():
        raise FileNotFoundError(
            f"{path}가 없습니다. ensure_pipeline('{label_version}')을 먼저 실행하세요"
        )
    return pd.read_csv(path, encoding="utf-8")


def load_folds(split_name: str, fold: int | None = None) -> pd.DataFrame:
    """fold_k.csv(사진마다 한 행, split 열이 train/val)를 읽는다. fold가 None이면 모두 합친다."""
    split_dir = PROCESSED_DIR / "splits" / split_name
    folds = range(N_FOLDS) if fold is None else [fold]
    paths = [split_dir / f"fold_{k}.csv" for k in folds]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{missing[0]}가 없습니다. ensure_pipeline()을 먼저 실행하세요")
    return pd.concat([pd.read_csv(p, encoding="utf-8") for p in paths], ignore_index=True)


def load_fold_labels(split_name: str, fold: int) -> pd.DataFrame:
    """분할 이름의 라벨 버전으로 라벨을 읽고 그 fold의 split(train/val) 열을 붙인다."""
    from data_process.split import parse_split_name

    label_version = parse_split_name(split_name).label_version
    folds = load_folds(split_name, fold)[["file_name", "split"]]
    return load_labels(label_version).merge(folds, on="file_name", how="left", validate="m:1")


def image_path(file_name: str, label_version: str = "v1") -> Path:
    """학습 이미지 경로. data_main이 만든 YOLO 폴더의 링크를 가리킨다."""
    return yolo_dir(label_version) / "images" / "train" / file_name

"""일시적인 이미지 읽기 실패를 복구하는 데이터 로더 보조 함수."""

import time
import warnings
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils.patches import imread as ultralytics_imread


def read_image_with_retry(filename: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """기본 이미지 읽기 실패 시 일반 파일 읽기로 최대 세 번 재시도한다."""
    path = Path(filename)
    errors = []
    for attempt in range(3):
        try:
            image = ultralytics_imread(str(path), flags=flags)
            if image is not None:
                return image
        except (OSError, cv2.error) as error:
            errors.append(f"기본 로더: {error}")
        try:
            contents = path.read_bytes()
            if not contents:
                raise OSError("이미지 바이트가 비어 있습니다.")
            image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), flags)
            if image is not None:
                if image.dtype == np.uint16 and (
                    flags == cv2.IMREAD_UNCHANGED or not flags & cv2.IMREAD_ANYDEPTH
                ):
                    image = (image >> 8).astype(np.uint8)
                if image.ndim == 2:
                    image = image[..., None]
                warnings.warn(f"일반 파일 읽기로 이미지 복구: {path}", stacklevel=2)
                return image
            errors.append(f"재시도 {attempt + 1}: {len(contents)} bytes 디코딩 실패")
        except (OSError, cv2.error) as error:
            errors.append(f"재시도 {attempt + 1}: {error}")
        if attempt < 2:
            time.sleep(0.1)
    if not path.is_file():
        raise FileNotFoundError(f"이미지 또는 링크 대상이 없습니다: {path} → {path.resolve()}")
    raise OSError(f"이미지를 반복해서 읽지 못했습니다: {path}. " + "; ".join(errors))


def install_image_read_retry() -> None:
    """현재 프로세스의 Ultralytics 데이터셋 로더에 읽기 복구를 적용한다."""
    from ultralytics.data import base

    base.imread = read_image_with_retry

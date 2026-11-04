"""공동 OCR의 ROIAlign을 컴파일러 없이 결정론적 PyTorch 연산으로 실행한다."""

from importlib import import_module
from inspect import unwrap

from torch import Tensor


def eager_roi_align(feature: Tensor, rois: Tensor, height: int, width: int) -> Tensor:
    """torchvision의 동일 ROIAlign 구현에서 자동 컴파일 래퍼만 제외한다."""
    module = import_module("torchvision.ops.roi_align")
    implementation = getattr(module, "_roi_align", None)
    if implementation is None or not hasattr(implementation, "__wrapped__"):
        raise RuntimeError("현재 torchvision에서 공동 OCR의 비컴파일 ROIAlign을 찾지 못했습니다.")
    # 전역 deterministic 설정이나 torchvision 함수를 바꾸지 않는다.
    return unwrap(implementation)(feature, rois, 1.0, height, width, 2, True)

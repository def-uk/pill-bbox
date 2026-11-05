"""검증 분할 없이 전체 데이터로 학습하는 노트북 전용 YOLO trainer."""

import shutil
from pathlib import Path

from torch.utils.data import DataLoader
from ultralytics.models.yolo.detect import DetectionTrainer


class FullTrainTrainer(DetectionTrainer):
    """마지막 epoch와 종료 후에도 검증을 생략하고 last.pt를 제출에 사용한다."""

    def get_dataloader(
        self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"
    ) -> DataLoader:
        """검증용 데이터를 따로 읽지 않고 부모 초기화에 필요한 참조만 제공한다."""
        if mode == "val":
            return self.train_loader
        return super().get_dataloader(dataset_path, batch_size, rank, mode)

    def _setup_train(self) -> None:
        """검증 지표를 초기 기록에서 제거한다."""
        super()._setup_train()
        self.metrics = {}

    def validate(self) -> tuple[dict, None]:
        """부모의 마지막 epoch 강제 검증도 실제 추론 없이 건너뛴다."""
        return {}, None

    def final_eval(self) -> None:
        """학습 종료 후 자동 best 검증을 생략한다."""

    def save_model(self) -> bool:
        """기존 완료 epoch 기준 주기 체크포인트 저장을 유지한다."""
        period = self.save_period
        self.save_period = -1
        try:
            saved = super().save_model()
        finally:
            self.save_period = period
        if saved and period > 0 and (self.epoch + 1) % period == 0:
            shutil.copyfile(self.last, Path(self.wdir) / f"epoch{self.epoch + 1}.pt")
        return saved

"""공유 8:2 검증 분할을 사용하는 노트북 YOLO trainer."""

import shutil
from pathlib import Path

from models.train import Map7595Trainer


class HoldoutTrainer(Map7595Trainer):
    """mAP75-95로 best.pt를 고르고 완료 epoch 기준 주기 저장을 유지한다."""

    def save_model(self) -> bool:
        """완료한 epoch 수에 맞춰 추가 체크포인트를 저장한다."""
        period = self.save_period
        self.save_period = -1
        try:
            saved = super().save_model()
        finally:
            self.save_period = period
        if saved and period > 0 and (self.epoch + 1) % period == 0:
            shutil.copyfile(self.last, Path(self.wdir) / f"epoch{self.epoch + 1}.pt")
        return saved

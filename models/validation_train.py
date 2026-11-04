"""공유 8:2 검증 분할을 사용하는 노트북 YOLO trainer."""

import shutil
from pathlib import Path

from models.train import Map7595Trainer


class HoldoutTrainer(Map7595Trainer):
    """mAP75-95로 best.pt를 고르고 완료 epoch 기준 주기 저장을 유지한다."""

    def save_model(self) -> bool:
        """임시 파일로 저장한 뒤 교체해 중단 시 기존 체크포인트를 보존한다."""
        period = self.save_period
        last, best = self.last, self.best
        pending_last = last.with_name("last.pending.pt")
        pending_best = best.with_name("best.pending.pt")
        pending_best.unlink(missing_ok=True)
        self.save_period = -1
        self.last, self.best = pending_last, pending_best
        try:
            saved = super().save_model()
        finally:
            self.save_period = period
            self.last, self.best = last, best
        if saved:
            pending_last.replace(last)
            if pending_best.is_file():
                pending_best.replace(best)
        if saved and period > 0 and (self.epoch + 1) % period == 0:
            shutil.copyfile(self.last, Path(self.wdir) / f"epoch{self.epoch + 1}.pt")
        return saved

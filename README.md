<div align="center">

<img src="docs/assets/logo.png" width="140" alt="알약몬 로고"/>

# 알약몬 <sub><sup>(구 먹었약)</sup></sub>

**알약 사진 한 장으로 복용을 기록하고, 꾸준한 기록을 캐릭터 성장과 몬스터 퇴치로 바꾸는 서비스**

이 저장소는 알약몬의 알약 인식 모델입니다. 알약 사진에서 알약마다 위치(bbox)와 종류(56종)를 찾습니다.<br/>
서비스 앱은 [pill-bbox-front](https://github.com/def-uk/pill-bbox-front)에 있습니다.

[![App](https://img.shields.io/badge/app-pill--bbox--front-452AF7?logo=flutter&logoColor=white)](https://github.com/def-uk/pill-bbox-front)
[![Python](https://img.shields.io/badge/Python-3.12-452AF7?logo=python&logoColor=white)](pyproject.toml)
[![uv](https://img.shields.io/badge/uv-managed-452AF7?logo=astral&logoColor=white)](https://docs.astral.sh/uv/)
[![Model](https://img.shields.io/badge/model-YOLO11m-452AF7)](#결과)
[![Kaggle](https://img.shields.io/badge/Kaggle%20mAP%400.75%3A0.95-0.6306-452AF7?logo=kaggle&logoColor=white)](#결과)
[![Notion](https://img.shields.io/badge/%EC%A7%84%ED%96%89%20%EA%B8%B0%EB%A1%9D-Notion-000000?logo=notion&logoColor=white)](https://app.notion.com/p/4fbcef24bb194f369f6751ae5a0b6e61?source=copy_link)

[문제 정의](#문제-정의) · [서비스 구상](#서비스-구상) · [데이터](#데이터) · [접근 방법](#접근-방법) · [실험](#실험) · [결과](#결과) · [팀](#팀) · [시작하기](#시작하기) · [개발 가이드](#개발-가이드) · [폴더 구조](#폴더-구조)

</div>

<br/>

| 🎯 과제 | 🗂️ 데이터 | 🏆 최종 모델 | 📈 최고 Kaggle 점수 |
|:---:|:---:|:---:|:---:|
| 알약 bbox + 56종 분류 | 대회 232장 + AI Hub 외부 데이터 | **YOLO11m** (2026-11-04 결정) | **0.63060** (mAP@[0.75:0.95]) |

## 문제 정의

> **어떻게 하면 1020세대가 복용 기록을 꾸준히 하게 할까?**

1020세대도 비타민·영양제를 매일 챙기고, 감기나 피부 질환으로 며칠에서 몇 주씩 처방약을 먹습니다.
어려운 것은 약을 먹는 일보다 **먹었는지 기록하고 정해진 기간을 끝까지 지키는 일**입니다.
기록할 이유가 없으면 며칠 만에 그만두고, 처방약은 증상이 나아지면 남은 기간을 빠뜨리기 쉽습니다.

알약몬은 알약 사진을 찍으면 어떤 약인지 알아보고 찍은 시각에 먹은 것으로 기록합니다.
그리고 그 기록을 **캐릭터 성장과 질병 몬스터 퇴치**로 보여 줘서, 기록을 이어 갈 이유를 만듭니다.

| 누가 | 언제 쓰나 |
|---|---|
| 비타민·영양제를 챙겨 먹는 1020 | 매일 먹은 기록으로 내 캐릭터를 키우고 꾸밀 때 |
| 처방약을 받은 1020 | 처방된 기간을 끝까지 지켜 질병 몬스터를 물리칠 때 |

알림과 체크 위주의 복약 관리 앱과 다른 점:

| | 알림·체크 위주의 복약 앱 | 알약몬 |
|---|---|---|
| 기록 방법 | 먹은 약을 직접 골라 체크 | 알약 사진 한 장(검출 모델), 처방전 스캔(OCR) |
| 계속할 이유 | 정해진 시각의 알림 | 캐릭터 성장, 몬스터 퇴치, 유니크 뽑기, 컬렉션 |

정한 원칙:

- **정확도가 우선입니다.** 약을 잘못 알려 주면 위험하므로 박스와 클래스가 정확해야 합니다.
  그래서 대회 지표도 박스가 꼭 맞아야 점수가 나는 mAP@[0.75:0.95]입니다.
- **속도도 봅니다.** 휴대폰 앱으로 낼 서비스라 QR 코드처럼 바로 결과가 나와야 합니다. 같은 정확도라면 one-stage 모델을 고릅니다.
- **약통이 아니라 알약을 봅니다.** 약통 라벨은 믿을 수 없습니다(타이레놀 통에 감기약을 넣어 둘 수도 있습니다).
- 깨지거나 오염된 약은 확신도가 낮으면 "확인할 수 없는 약"으로 처리하는 방향을 논의했습니다.

| 입력 | 출력 |
|---|---|
| 알약 사진 (한 장에 알약 최대 4개) | 알약마다 bbox, 클래스(약 이름), 확신도 |

## 서비스 구상

이 저장소의 검출 모델이 "무슨 약을 먹었는지"를 알아내면, 알약몬은 그 기록을 약 종류에 따라 두 가지 성장 루프로 연결합니다.

```mermaid
flowchart LR
    A[알약 사진] --> C[복용 기록<br/>하루 기준량까지만]
    B[처방전 스캔<br/>OCR] --> S[복용 일정 등록] --> C
    C -->|비타민·영양제| D[내 캐릭터 성장]
    C -->|치료약| E[질병 몬스터 체력 감소]
    D -->|기록을 이어 가면| G[유니크 뽑기<br/>스킨·액세서리]
    E -->|복용 기간을 마치면| H[몬스터 퇴치<br/>뽑기 + 몬스터 펫]
    G --> K[캐릭터 꾸미기·컬렉션 전시]
    H --> K
```

| 기능 | 내용 |
|---|---|
| 처방전 스캔으로 일정 등록 | OCR로 약 이름, 복용 기간, 하루 복용 횟수, 1회 알 수를 읽고 사용자가 확인·수정하면 복용 일정이 됩니다. |
| 비타민·영양제 → 캐릭터 성장 | 일정대로 기록하면 캐릭터가 자라고, 기록을 이어 가면 유니크 뽑기로 스킨·액세서리를 얻습니다. |
| 치료약 → 질병 몬스터 퇴치 | 약을 등록하면 질병 몬스터가 나타납니다. 체력은 처방 기간에 맞추고, 하루치를 다 먹을 때마다 줄어듭니다. |
| 모으고 보여 주는 보상 | 유니크 스킨과 몬스터 펫을 장착하거나 컬렉션에 전시해 "나도 갖고 싶다"는 마음이 들게 합니다. |
| 하루 기준량까지만 보상 | 치료약은 처방된 하루 복용량, 영양제는 제품의 하루 권장량까지만 반영합니다. 넘긴 기록에는 성장·체력 감소·뽑기가 없습니다. |

### 서비스 앱

사진 한 장으로 복용을 기록하는 흐름은 [pill-bbox-front](https://github.com/def-uk/pill-bbox-front)에 Flutter 앱으로 구현되어 있습니다.

| 항목 | 내용 |
|---|---|
| 하는 일 | 알약 사진에서 약 종류와 개수를 찾아 복용 기록으로 남김. 처음 보는 약은 저장할 때 등록 |
| 추론 | 서버 없이 기기 안에서 실행 (웹 ONNX, Android TFLite, iOS Core ML) |
| 쓰는 모델 | `combofix-v1-fold1-w8` (`yolo11m`). 대회 56종에 AI Hub 40종을 더한 96종, 가중치 int8(80MB → 20MB) |

> [!NOTE]
> 캐릭터 성장, 몬스터 퇴치, 처방전 스캔은 아직 확장 제안 단계입니다. 기능별 상세 설명은 [docs/SERVICE_PLAN.md](docs/SERVICE_PLAN.md)에 있습니다.

## 데이터

| 구분 | 내용 |
|---|---|
| 대회 데이터 | train 이미지 232장, annotation JSON 763개(bbox 763개), test 이미지 842장, 클래스 56종 |
| 외부 데이터 | [AI Hub 경구약제 이미지 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=576)의 조합 데이터. train에만 추가하고 검증은 대회 사진으로만 함 |
| 평가 지표 | mAP@[0.75:0.95] (IoU 0.75–0.95 다섯 기준의 AP 평균). 코드에서는 `mAP75-95` |

EDA로 찾은 문제와 처리:

- **라벨 오류**: 누락 라벨 8개를 추가하고 위치가 틀린 박스 5개와 잘린 캡슐 박스를 고쳐 기본 라벨 `v1`(bbox 771개)을 만들었습니다.
  원본 JSON은 고치지 않고 보정 기록(`data/processed/corrections.csv`)으로 적용합니다.
- **누수**: 같은 약 조합을 각도만 바꿔 찍은 사진이 많아, 무작위로 나누면 검증 점수가 부풀려집니다.
  조합 단위(`combo_v1_s42`)와 촬영 세팅 단위(`setup30fix_v1_s42`) 5 fold 분할을 만들고, bbox 5개 이하의 희귀 약이 든 조합은 train에 고정했습니다.
- **클래스 불균형**: 희귀 약 때문에 AI Hub 외부 데이터를 가져왔습니다. 대회 56종과 겹치는 클래스는 클래스당 bbox 157개까지 채우고,
  박스가 맞지 않는 사진은 사람이 검수해 빼거나 고쳤습니다(2차 EDA). 대회에 없는 약은 새 클래스로 학습하고 평가에서는 뺍니다.

자세한 데이터 구조는 [data/README.md](data/README.md), 만드는 방법은 [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md)를 참고하세요.

## 접근 방법

```mermaid
flowchart LR
    A[원본 JSON·이미지] --> B[라벨 보정<br/>v1]
    B --> C[5 fold 분할<br/>누수 검사]
    C --> D[YOLO 변환<br/>data.yaml]
    E[AI Hub 외부 데이터<br/>선별·검수] --> D
    D --> F[학습<br/>YOLO11m · YOLO26m · Faster R-CNN]
    F --> G[평가<br/>mAP75-95 · P/R · 추론 속도]
    G --> H[Kaggle 제출 csv]
```

- **후보 모델**: one-stage(YOLO11m, YOLO26m)와 two-stage(Faster R-CNN)를 같은 데이터·같은 분할로 비교했습니다.
  YOLO26m은 NMS 없이 바로 박스를 내는 출력층(end-to-end)도 함께 평가했습니다.
- **공통 베이스라인**: 같은 하이퍼파라미터, 같은 분할, 같은 평가 코드로 맞춘 뒤 실험마다 변수 하나만 바꿨습니다.
- **평가**: mAP75-95를 주 지표로 `best.pt`와 early stopping을 정하고, 정밀도·재현율·추론 속도·혼동행렬·실패 사례를 함께 봤습니다.
  성능이 오르든 내리든 모든 실험을 문서화했습니다.

공통 베이스라인 설정:

| 항목 | 값 |
|---|---|
| 이미지 크기 / 배치 | 640 / 16 (GPU 메모리에 따라 8·4) |
| epochs / patience | 100 / 10 |
| 옵티마이저 | MuSGD, lr0 0.001, warmup 3 에폭 |
| 증강 (`on`) | 뒤집기 끔, 회전 10°, 색조 0.005 (알약 각인과 색이 클래스를 가르므로 약하게) |
| seed | 42 |
| 평가 | conf 0.001, max_det 4 |

## 실험

실험마다 베이스라인 대비 한 가지만 바꿨습니다. 전체 목록과 그래프는 [Notion 실험 페이지](https://app.notion.com/p/4fbcef24bb194f369f6751ae5a0b6e61?source=copy_link)에 있습니다.

### 1. 베이스라인 및 아키텍처 비교

one-stage와 two-stage 후보를 같은 데이터·분할·평가 코드로 맞춰 비교했습니다.

| 실험 | 모델 | 날짜 |
|---|---|---|
| 베이스라인 구축 (분할, 평가 지표, training loop, 혼동행렬, 실패 사례 시각화, 제출 csv) | YOLO11m, YOLO26m, Faster R-CNN | 10.30 |
| 100 에폭 학습 | YOLO11m, Faster R-CNN | 10.31 |

### 2. 데이터 품질 개선

희귀 클래스를 채우려고 외부 데이터를 더하고, 박스가 맞지 않는 사진은 걸러 냈습니다.

| 실험 | 모델 | 날짜 |
|---|---|---|
| AI Hub 외부 데이터 추가 | YOLO11m, YOLO26m, Faster R-CNN | 11.01 |
| 2차 EDA로 선별·검수한 외부 데이터 | YOLO11m, YOLO26m | 11.03 |

### 3. 증강 기법 비교

알약은 각인과 색이 클래스를 가르므로 어떤 증강이 도움이 되고 해가 되는지 하나씩 확인했습니다.

| 실험 | 모델 | 날짜 |
|---|---|---|
| 모자이크 증강 켜기·끄기 | YOLO11m, YOLO26m, Faster R-CNN | 11.01–11.02 |
| 회전 증강 | YOLO11m, Faster R-CNN | 11.02 |
| Copy-paste 증강 (정답 bbox를 잘라 다른 사진에 붙이는 방식을 직접 구현) | YOLO11m, YOLO26m, Faster R-CNN | 11.03 |
| 최고 모델에서 모자이크 끄기, 채도·색조·밝기 증강 줄이기 | YOLO11m | 11.04 |

### 4. 학습 방식 변경

| 실험 | 모델 | 날짜 |
|---|---|---|
| 각인 OCR을 함께 학습해 재분류 | YOLO11m, YOLO26m | 11.03 |

### 5. 최종 모델

| 실험 | 모델 | 날짜 |
|---|---|---|
| 검증 없이 전체 데이터로 최종 모델 학습 (예정) | YOLO11m | 11.04 |

실험 코드는 실험별 브랜치(`exp/<모델>_<실험>`)와 [notebooks/](notebooks/)에 있습니다.

## 결과

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/results_dark.svg">
  <img src="docs/assets/results_light.svg" alt="모델·설정별 Kaggle mAP@[0.75:0.95]. 베이스라인은 YOLO11m 0.299, YOLO26m 0.179, Faster R-CNN 0.259. 외부 데이터를 더한 YOLO11m 0.631, 외부 데이터와 copy-paste를 쓴 YOLO26m 0.607." width="820">
</picture>

<details>
<summary>표로 보기</summary>

| 모델 | 설정 | Kaggle mAP@[0.75:0.95] |
|---|---|---|
| YOLO11m | 베이스라인 (30 에폭, 외부 데이터 없음) | 0.29946 |
| YOLO26m | 베이스라인 (30 에폭, 외부 데이터 없음) | 0.17877 |
| Faster R-CNN | 베이스라인 (30 에폭, 외부 데이터 없음) | 0.25918 |
| YOLO11m | 외부 데이터, 외부 클래스까지 학습 | 0.63060 |
| YOLO26m | 외부 데이터 + copy-paste | 0.607 |
| **YOLO11m** | **최종 모델** (전체 데이터 학습 후 기입) | |

</details>

<!-- 점수를 바꾸면 docs/assets/make_results_chart.py의 RESULTS도 고치고 다시 실행하세요 -->

> [!IMPORTANT]
> **모델 선택: YOLO11m**
>
> - Kaggle 점수가 세 후보 중 가장 높았습니다.
> - 휴대폰에서 쓸 서비스라 one-stage 모델의 속도가 이점입니다.
> - Faster R-CNN은 증강을 YOLO 기준으로 맞춰 실험해 손해를 봤을 수 있습니다. 시간 안에 다시 맞추기 어려워 후보에서 뺐습니다.

### 알게 된 점

- 각인 OCR을 함께 학습하면 오히려 점수가 떨어졌습니다. 원인은 분석 중입니다.
- YOLO의 내장 `copy_paste`는 세그멘테이션 마스크용이라 bbox 데이터에서는 동작하지 않아, bbox copy-paste를 직접 구현했습니다.

> [!NOTE]
> 검증 mAP75-95(0.9대)가 Kaggle 점수(0.6대)보다 크게 높았습니다. 조합 단위 분할에도 촬영 세팅이 같은 사진이 train과 검증에 함께 남아 있기 때문으로 보고,
> 모델 선택은 Kaggle 점수로 했습니다. 시간 때문에 fold 0 하나로 실험했으므로 5 fold 평균 비교는 남은 과제입니다.

### 향후 계획

- 증강을 Faster R-CNN에 맞춰 다시 실험합니다. 약은 정확도가 우선이라 two-stage 모델의 가능성을 더 확인할 가치가 있습니다.
- 앱 사용자가 찍은 사진을 모아 정제한 뒤 학습 데이터로 씁니다.
- 앙상블(WBF), TTA를 시도합니다.

## 팀

<table>
  <tr>
    <td align="center" valign="top" width="25%">
      <a href="https://github.com/def-uk"><img src="https://github.com/def-uk.png" width="100" alt="동욱"/><br/><b>동욱</b></a><br/>
      <sub><a href="https://github.com/def-uk">@def-uk</a></sub><br/><br/>
      <sub>데이터 파이프라인·학습·평가 코드, Faster R-CNN 베이스라인, YOLO26m 실험, 프론트엔드 데모, README</sub>
    </td>
    <td align="center" valign="top" width="25%">
      <a href="https://github.com/gwon753-dev"><img src="https://github.com/gwon753-dev.png" width="100" alt="영현"/><br/><b>영현</b></a><br/>
      <sub><a href="https://github.com/gwon753-dev">@gwon753-dev</a></sub><br/><br/>
      <sub>라벨 보정, YOLO11m 베이스라인·외부 데이터·모자이크·OCR 실험, 프로젝트 문서화</sub>
    </td>
    <td align="center" valign="top" width="25%">
      <a href="https://github.com/JJAM99"><img src="https://github.com/JJAM99.png" width="100" alt="재민"/><br/><b>재민</b></a><br/>
      <sub><a href="https://github.com/JJAM99">@JJAM99</a></sub><br/><br/>
      <sub>YOLO11m 베이스라인·증강 실험, 외부 데이터 1·2차 EDA·전처리</sub>
    </td>
    <td align="center" valign="top" width="25%">
      <a href="https://github.com/younghoon1215"><img src="https://github.com/younghoon1215.png" width="100" alt="영훈"/><br/><b>영훈</b></a><br/>
      <sub><a href="https://github.com/younghoon1215">@younghoon1215</a></sub><br/><br/>
      <sub>YOLO26m 베이스라인, Faster R-CNN 실험, 발표 자료</sub>
    </td>
  </tr>
</table>

<!-- 맡은 일이 다르면 고쳐 주세요 -->

## 시작하기

[uv](https://docs.astral.sh/uv/)로 Python 3.12와 패키지를 맞춥니다.

```bash
git clone https://github.com/def-uk/pill-bbox.git && cd pill-bbox
uv sync                                                   # 환경 만들기
git config core.hooksPath .githooks                       # main 직접 커밋 방지 hook
cp .env.example .env                                      # DATA_URL, WANDB_API_KEY 등 입력
uv run python -m data_process.data_main --label v1        # 원본 → 라벨 → 분할 → data.yaml
```

학습과 Kaggle 제출 csv 만들기:

```bash
# 동작 확인 (1 에폭, 데이터 10%, wandb 끔)
WANDB_MODE=disabled uv run python -m models.model_main --model yolo11m.pt \
    --split combo_v1_s42 --folds 0 --epochs 1 --fraction 0.1 --exp smoke

# 제출 csv
uv run python -m models.evaluate --weights <best.pt 경로> --exp <실험 이름>
```

본 학습은 GPU(Colab, Kaggle)에서 합니다. 인자와 결과 위치는 [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md)를 참고하세요.

## 개발 가이드

### 치트시트

| 상황 | 명령 |
|---|---|
| pull 받은 뒤 | `uv sync` |
| 학습 데이터 만들기 | `uv run python -m data_process.data_main --label v1` |
| 노트북 열기 | `uv run jupyter lab` |
| 패키지 추가 / 제거 | `uv add <패키지>` / `uv remove <패키지>` |
| 개발용 패키지 추가 | `uv add --dev <패키지>` |
| 커밋 전 코드 정리 | `uv run ruff format .` → `uv run ruff check . --fix` |
| Mac에서 MPS 에러 | `PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python <파일>` |
| wandb 끄고 실행 | `WANDB_MODE=disabled uv run python <파일>` |

작업 흐름:

```bash
git switch main && git pull && uv sync      # 최신 상태로 맞추기
git switch -c feat/<짧은-설명>                # 브랜치 만들기
# ... 작업 ...
uv run ruff format . && uv run ruff check . --fix
git add <파일> && git commit -m "feat: <요약>"
git push -u origin feat/<짧은-설명>
gh pr create --base main                     # PR 본문은 템플릿을 채움
```

- 패키지는 `uv add`로만 추가하고 `pyproject.toml`과 `uv.lock`을 함께 커밋합니다. `pip install`은 쓰지 않습니다.
- 노트북은 출력을 지운 뒤 커밋합니다.

### 환경

| OS | PyTorch | 장치 |
|---|---|---|
| macOS | PyPI 기본 빌드 | `mps` (Apple GPU) |
| Windows / Linux | CUDA 12.6 빌드 | `cuda` (NVIDIA GPU) |

장치는 `src/config.py`의 `get_device()`가 고릅니다(cuda → mps → cpu). 코드에 `"cuda"`, `"mps"`를 직접 쓰지 않습니다.

### 실험 기록 (wandb)

`.env`에 [API 키](https://wandb.ai/authorize)와 팀 이름(`WANDB_ENTITY`)을 넣고, 학습 전에 `init_wandb()`를 부릅니다.
ultralytics 전역 설정은 바꾸지 않으므로 이 프로젝트 밖의 YOLO 학습에는 영향이 없습니다.

```python
from ultralytics import YOLO

from src.config import get_device, init_wandb

model = YOLO("yolo11m.pt")
init_wandb(model, run_name="yolo11m-baseline", config={"seed": 42})
model.train(data="data.yaml", epochs=100, device=get_device())
```

인터넷 없이 기록만 남기려면 `WANDB_MODE=offline`, 나중에 `uv run wandb sync wandb/offline-run-*`로 올립니다.

### 규칙 문서

| 문서 | 내용 |
|---|---|
| [docs/CODE_CONVENTION.md](docs/CODE_CONVENTION.md) | 코드 컨벤션 (포맷, 이름, import, 경로·장치) |
| [docs/GIT_CONVENTION.md](docs/GIT_CONVENTION.md) | Git 컨벤션 (커밋 메시지 `<type>: <요약>`, 브랜치 `<type>/<짧은-설명>`, PR) |
| [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) | 데이터 파이프라인 (라벨 버전, 분할, 학습·평가 명령, 여백 검수) |
| [data/README.md](data/README.md) | 데이터 폴더 구조, 외부 데이터, annotation JSON 구조 |

## 폴더 구조

```
pill-bbox/
├── data/               # 데이터 (이미지·annotation은 커밋 안 함)
├── data_process/       # 데이터 파이프라인 (라벨 보정, 분할, YOLO 변환, AI Hub 읽기)
├── models/             # 학습(train.py, model_main.py), 평가·제출(evaluate.py), 튜닝
├── notebooks/          # EDA, 모델별 실험 노트북
├── src/                # 공통 설정 (config.py)
├── docs/               # 컨벤션, 데이터 파이프라인 문서
├── team_test/          # 팀원 Git 연습용
├── main.py
├── pyproject.toml
└── uv.lock
```

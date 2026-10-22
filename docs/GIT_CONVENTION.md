# Git 컨벤션

## 처음 한 번: git hook 켜기

`main`에 직접 커밋·push하는 실수를 막는 hook이 `.githooks/`에 있습니다. clone한 뒤 한 번만 실행하세요.

```bash
git config core.hooksPath .githooks
```

| hook | 막는 것 |
|---|---|
| `pre-commit` | `main` 브랜치에서 커밋 |
| `pre-push` | 원격 `main`으로 push (삭제·강제 push 포함) |

## 커밋 메시지

### 형식

```
<type>: <요약>

본문(선택)
```

- 요약은 한국어로, 50자 이내로, 끝에 마침표 없이 씁니다.
- 관련 이슈가 있으면 끝에 `#이슈번호`를 붙입니다.
- 커밋 하나에는 한 가지 변경만 담습니다.

### type

| type | 언제 | 예시 |
|---|---|---|
| `feat` | 새 기능 | `feat: YOLO 라벨 변환 스크립트 추가` |
| `fix` | 버그 수정 | `fix: bbox 좌표가 이미지 밖으로 나가는 문제 수정` |
| `exp` | 실험 설정, 하이퍼파라미터 변경 | `exp: yolov8s, imgsz 640으로 학습` |
| `data` | 데이터 전처리, 분할 방식 변경 | `data: train/val 분할 비율 8:2로 변경` |
| `refactor` | 동작은 그대로, 구조만 정리 | `refactor: 경로 설정을 config.py로 이동` |
| `docs` | 문서 | `docs: README에 실행 방법 추가` |
| `chore` | 패키지, 설정, 기타 | `chore: optuna 의존성 추가` |

### 예시

```
exp: yolov8m 기본 설정으로 50 epoch 학습

yolov8s 대비 작은 알약 검출이 부족해서 모델 크기를 키움.
val mAP50: 0.81 -> 0.86

#12
```

## 브랜치

`main`에 바로 push하지 않고, 브랜치를 만들어 PR로 합칩니다.

```
<type>/<짧은-설명>      예: feat/label-converter, exp/yolov8m-baseline
```

## PR 올리기

브랜치를 push한 뒤 GitHub CLI(`gh`)로 PR을 만듭니다.

```bash
# 처음 한 번: 설치와 로그인
brew install gh                    # macOS
winget install --id GitHub.cli     # Windows
sudo apt install gh                # Linux/Ubuntu
gh auth login

# PR 만들기
git push -u origin feat/<짧은-설명>
gh pr create --base main           # 제목·본문을 묻고, 본문에는 PR 템플릿이 채워짐
gh pr create --base main --web     # 브라우저에서 작성하고 싶을 때

# PR 확인
gh pr status                       # 내 PR 상태
gh pr view --web                   # 현재 브랜치의 PR을 브라우저로 열기
```

- PR 제목은 커밋 형식과 같이 `<type>: <요약>`으로 씁니다.
- 본문은 [PR 템플릿](../.github/PULL_REQUEST_TEMPLATE.md)을 채웁니다. `--fill`을 쓰면 템플릿이 빠지므로 쓰지 않습니다.
- `gh`를 쓰지 않는다면 push 후 GitHub 저장소 페이지에 뜨는 **Compare & pull request** 버튼을 눌러도 됩니다.

## 최신 버전 받기

PR이 머지된 뒤 새 작업을 시작할 때는 `main`을 최신으로 받고 새 브랜치를 만듭니다. 머지된 브랜치에서 이어서 작업하지 않습니다.

```bash
git switch main
git pull origin main
git switch -c feat/<짧은-설명>
```

작업 중인 브랜치에 `main`의 최신 변경을 반영할 때는 브랜치를 옮기지 않고 그대로 받습니다.

```bash
git pull origin main               # main의 변경을 현재 브랜치에 merge

# 커밋하지 않은 작업이 있어 pull이 막힐 때
git stash                          # 작업 중인 변경을 임시 저장
git pull origin main
git stash pop                      # 작업 내용 되돌리기
```

- 충돌이 나면 해당 파일의 `<<<<<<<` ~ `>>>>>>>` 구간을 정리하고 `git add <파일>` 후 커밋합니다.
- `git stash pop`에서 충돌이 나면 stash가 남아 있으므로, 정리한 뒤 `git stash drop`으로 지웁니다.

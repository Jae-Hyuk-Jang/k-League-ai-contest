# KLeagueAI Open Track 1 제출물 (v7)

KLeagueAI Open Track 1(패스 종료 좌표 예측) 제출 코드입니다.

- `kleague_v7_core.py` : 피처/데이터셋/모델/학습/추론 로직이 모두 들어있는 코어 모듈
- `train.py` : 5-fold CV 학습 재현용 엔트리포인트
- `inference.py` : 저장된 가중치/메타로 테스트 에피소드 추론 후 제출 CSV 생성

본 제출물은 **상대 경로(`./data`, `./weights`, `./output`) 기반**으로 동작하도록 구성되어 있습니다.

---

## 1) 폴더 구조

아래 구조를 권장합니다.

```
submit/
  data/
    train.csv
    test.csv
    sample_submission.csv
    test/                # test episode CSV들
      *.csv
    (선택) match_info.csv
    (선택) data_description.xlsx
  weights/
    fold0.pt
    fold1.pt
    fold2.pt
    fold3.pt
    fold4.pt
    meta_fold0.pkl
    meta_fold1.pkl
    meta_fold2.pkl
    meta_fold3.pkl
    meta_fold4.pkl
    mappings.pkl         # 또는 mappings.json
    mappings.json
    cfg.json
    fold_scores.json     # (선택) 있으면 가중 앙상블에 사용
    env_info.json        # (학습 시) 환경 스냅샷
  output/
  kleague_v7_core.py
  train.py
  inference.py
  requirements.txt
  requirements_full.txt
```

> **(중요) test.csv / sample_submission.csv의 `path` 컬럼**
>
> `kleague_v7_core.py`의 추론 로직은 `sample_submission.csv`와 `test.csv`를 `game_episode`로 merge 한 뒤,
> 최종 데이터프레임에 **`path` 컬럼이 존재**한다고 가정합니다.
>
> - 일반적으로는 `test.csv`에 `path`가 있고, `sample_submission.csv`는 `game_episode, end_x, end_y`만 있는 형태가 안전합니다.
> - 만약 `test.csv`와 `sample_submission.csv` **둘 다 `path`가 있다면**, merge 후 `path_x/path_y`로 분리되어 코드가 깨질 수 있습니다.
>   이 경우에는 한쪽 파일의 `path`를 제거하거나 코드를 수정해야 합니다.

---

## 2) 설치

### (A) 최소 설치

```bash
pip install -r requirements.txt
```

### (B) 전체 스냅샷(감사용)

```bash
pip install -r requirements_full.txt
```

---

## 3) 추론 실행 (제출한 가중치가 존재하기 때문에 train.py 실행없이 동작할 수 있습니다)

**프로젝트 루트(`submit/`)에서 실행**을 권장합니다. (상대경로 사용)

```bash
python inference.py
```

- 기본 입력: `./data`
- 기본 가중치/메타: `./weights`
- 기본 출력: `./output`
- 기본 결과 파일명: `exp_dense_lstm_heatmap_v7_ema_softlabel_att_side_tta.csv`

경로를 바꾸고 싶으면:

```bash
python inference.py --data_root ./data --weights_dir ./weights --output_dir ./output
```

결과 파일명을 바꾸고 싶으면:

```bash
python inference.py --submission_filename my_submission.csv
```

### (중요) `path` → 실제 파일 경로 변환

추론 시 내부에서 아래 변환을 수행합니다.

- `path` 문자열에서 `"./"`를 제거하고
- `data_root`와 join

예) `path = "./test/00001.csv"` 이면
`<data_root>/test/00001.csv` 를 읽습니다.

> **주의:** 만약 `path`가 이미 `data/test/...` 형태라면 `data_root`와 합쳐져 중복 경로가 될 수 있습니다.
> 이 경우 episode 파일을 못 찾아 성능이 크게 떨어질 수 있으니, `path`는 보통 `./test/...` 또는 `test/...` 형태를 권장합니다.

### (중요) episode 파일 누락 시 동작

현재 `kleague_v7_core.py` 구현은 episode CSV를 찾지 못하면 해당 샘플을 **필드 중앙 좌표(52.5, 34.0)** 로 대체합니다.

- 제출 점수 급락을 막기 위해 **반드시 `data/test/*.csv` 존재 여부를 미리 확인**하는 것을 권장합니다.

---

## 4) 학습 재현 (5-fold CV)

```bash
python train.py
```

- `train.py`는 실행 시 **작업 디렉터리를 스크립트 위치로 고정**하여 상대경로가 안정적으로 동작합니다.
- 기본 설정은 원 학습과 유사하게 **batch 4096 / epoch 25 / AMP / EMA / mirror aug / last-pass finetune** 등을 사용합니다.
- GPU 메모리/시간 요구가 큽니다. (환경에 따라 batch/epoch 등을 줄여야 할 수 있습니다)

학습이 끝나면 `weights/` 아래에 아래 파일이 생성됩니다.

- `fold0.pt` ~ `fold4.pt`
- `meta_fold0.pkl` ~ `meta_fold4.pkl`
- `mappings.pkl` + `mappings.json`
- `cfg.json`
- `fold_scores.json`
- `env_info.json` (학습 환경 스냅샷)

### 기존 weights 덮어쓰기 방지

`train.py`는 기본적으로 `weights_dir`에 `fold*.pt`가 이미 있으면 중단합니다.
덮어쓰려면:

```bash
python train.py --overwrite
```

### 재현성 옵션

가능한 재현성을 높이려면(환경에 따라 에러가 날 수 있음):

```bash
python train.py --strict_deterministic --deterministic_warn_only --no_amp
```

- `--no_amp` : AMP 비활성화(재현성에 유리하지만 속도/메모리 손해)
- `--strict_deterministic` : PyTorch deterministic 알고리즘 강제(일부 CUDA 연산에서 예외 가능)
- `--deterministic_warn_only` : 가능하면 경고로 완화

> 참고: GPU/CUDA/cuDNN 버전이 달라지면 완전한 비트-단위 동일 재현은 보장하기 어렵습니다.

---

## 5) 모델/특징 요약

### 입력

- 에피소드 prefix 시퀀스(기본 최대 40 step)
  - 이동/시간/골대 관련 파생 피처(정규화)
  - 이벤트 타입 임베딩(`type_id`), 결과 임베딩(`result_id`)
  - 공격/수비(상대팀) indicator 임베딩(`att_side_id`)
- Heatmap(12x8 그리드, 8채널): prefix 이벤트의 start/end 분포 및 이동량을 요약
- 팀/선수/클러스터(episode cluster / player role / team style) 임베딩

### 출력(멀티태스크)

- coarse offset 회귀(정규화 dx,dy)
- zone 분류(6x4)
- fine cell 분류(24x16)
- fine residual 회귀(셀 중심 대비 잔차)
- coarse/fine mix gate

### 학습

- dense supervision: 각 에피소드에서 pass 이벤트를 타깃으로 샘플 생성(`train_only_pass_targets=True`)
- loss: offset MSE + end distance + zone CE + fine CE(spatial soft label) + residual L1 + coarse distance + gate reg
- mirror augmentation(Y flip) 및 stage-2 last-pass-only finetune(기본 heads-only)

### 추론

- fold 앙상블(기본 5개)
- fold_scores.json이 있으면 `1/(fold_score)` 기반 가중 앙상블
- fine head TOP-K 후보의 기대값으로 end 좌표 계산
- mirror TTA(옵션)로 평균

---

## 6) 흔한 에러 / 체크리스트

### (1) `FileNotFoundError: ... fold*.pt / meta_fold*.pkl`

- `weights/` 폴더에 필요한 아티팩트가 모두 있는지 확인하세요.

### (2) 추론 결과가 이상하게 나쁨(센터값이 많음)

- `data/test/*.csv` 파일이 실제로 존재하는지 확인
- `test.csv`의 `path` 값이 `./test/...` 형태인지 확인
- `data_root`를 올바르게 지정했는지 확인

### (3) `CUDA out of memory`

- 학습 시 `batch_size=4096`이 기본이라 GPU에 따라 OOM이 날 수 있습니다.
  - 해결: AMP 끄거나(`--no_amp`), cfg를 조정해 batch_size를 줄여야 합니다.

---

## 7) 실행 예시

Windows PowerShell:

```powershell
python -m pip install -r requirements.txt
python inference.py
```

Linux/macOS:

```bash
python3 -m pip install -r requirements.txt
python3 inference.py
```

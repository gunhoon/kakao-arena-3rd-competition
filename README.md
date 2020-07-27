# Kakao Arena 3rd Competition

Kakao Arena - Melon Playlist Continuation 대회 결과물


## 실행 환경

### 1) H/W 환경
**`CPU 8 Core, 메모리 16GB 이상을 권장함.`**

아래와 같은 환경에서의 학습 시간
* AWS EC2 c5.4xlarge(16 vCPU, 32GB 메모리) : 약 1.5 시간
* AWS EC2 c5.2xlarge(8 vCPU, 16GB 메모리) : 약 3 시간
* 2017년 맥북 프로(4 Core, 16GB 메모리) : 약 7~8 시간

### 2) S/W 환경
* python >= 3.6 (3.6, 3.7, 3.8에서 동작 확인함)

### 3) 필수 라이브러리
* tqdm
* numpy
* scipy
* pandas
* implicit
* khaiii


## 실행 방법

### 1) 데이터 다운로드

베이스라인 코드와 동일하게 res 디렉토리에 아래 4개 파일 다운로드 ([아레나 홈페이지 참조](https://arena.kakao.com/c/7/data))

```
.
└── /res
    ├── song_meta.json
    ├── test.json
    ├── train.json
    └── val.json
```

### 2) 학습

```bash
$> python train.py --question_fname=res/test.json
```
위와 같이 실행하면, 학습 모델이 `victoria.pkl` 파일 이름으로 저장된다.

파이널 제출용 결과(results.json)에 대한 모델(victoria.pkl) [다운로드](https://drive.google.com/file/d/1saL3KawG0sENvP1JhoQXHtbaIj4sYPKp/view?usp=sharing)

`**주의 사항**`
* implicit library 제한(random seed 없음)으로 제출용 모델과 정확히 동일한 모델을 학습으로 다시 만들 수 없음. 즉, 학습할 때 마다 약간씩 다른 모델이 생성됨. 그래서, 결과에도 +- 0.001 정도의 오차가 발생할 수 있음.
* 파이널 용으로 학습한 모델에 파이널이 아닌 다른 데이터(예를 들어, val.json)를 예측할 수 없음. 예측하려는 데이터를 학습 단계에서 `--question_fname=res/test.json` 파라미터로 넣어줘야 함.

### 3) 예측

```bash
$> python inference.py --question_fname=res/test.json
```
위와 같이 실행하면, `victoria.pkl` 파일을 로딩한 후, 결과 파일을 베이스라인 코드와 동일하게 `arena_data/results/results.json`에 저장한다.

```
.
├── victoria.pkl (train.py 실행 후 생성됨)
└── /arena_data
    └── /results
        └── results.json (inference.py 실행 후 생성됨)
```


## 소스 파일 구조

* baseline 디렉토리에는 베이스라인 코드가 포함되어 있고, 그 중에서 arena_util.py만 사용함.
* model 디렉토리의 victoria.py가 루트 코드이며, matrix_factorization.py와 fallback.py를 호출하는 구조임.
* matrix_factorization.py는 implicit library의 ALS를 사용하여, 메인 알고리즘인 MF를 구현한 코드임.
* fallback.py는 popularity 모델로, MF에서 추천하지 못한 playlist에 대한 backup 용도임.
* preference_data.py는 user x item matrix를 만들기 위해서, (user, item, preference) 형태의 table을 생성함.(matrix_factorization.py 에서 사용함.)
* tokenizer.py는 khaiii를 사용하여, playlist title의 형태소 분석하는 로직으로, preference_data.py에 의해 사용됨.
```
.
├── /baseline
│   └── arena_util.py
├── /model
│   ├── fallback.py
│   ├── matrix_factorization.py
│   ├── preference_data.py
│   ├── tokenizer.py
│   └── victoria.py
├── /res
├── inference.py
└── train.py
```


## 알고리즘 설명

implicit library를 사용하여 ALS 기반 MF(matrix factorization) 단일 모델임.

### 1) ALS 기반 MF

User X Item matrix를 생성할 때, 아래와 같이 Item에 여러 feature를 혼합하여 적절한 가중치를 줘서 matrix를 생성함.

* User는 playlist id
* Item은 song + tag + keyword(playlist title의 형태소분석) + 세부장르

### 2) Rule 기반 로직

위의 1)의 추천 리스트에서 몇 가지 Rule 기반 로직이 들어감.

* playlist에 동일 앨범의 곡이 특정 갯수(5개) 이상이고, 그 앨범에 수록된 다른 곡들이 추천 리스트에 포함되어 있다면, 그 곡들의 priority를 높임.
* playlist에 동일 아티스트의 곡이 특정 갯수(10개, 20개) 이상이고, 그 아티스트의 다른 곡들이 추천 리스트에 포함되어 있다면, 그 곡들의 priority를 높임.

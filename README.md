# [2025-1] 데이터사이언스 장기 프로젝트 — 추천 시스템 (GC-MC)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#라이선스)

영화 추천 문제를 그래프 기반 행렬 보완(Graph Convolutional Matrix Completion, GC‑MC)으로 풀어보는 실습형 프로젝트입니다. MovieLens 스타일의 사용자‑아이템 평점 데이터를 사용해 그래프 합성곱 인코더와 이중선형 디코더로 평점을 예측합니다.

> [!NOTE]
> 기본 모델은 PyTorch 기반 GC‑MC입니다. GPU가 있으면 학습 시간이 크게 단축됩니다. PyTorch가 없으면 실행이 제한됩니다.

## 프로젝트 구조

```
.
├── dataset/                  # MovieLens 스타일 데이터 (u.data, u.item, u.user, ...)
├── test/                     # 예시 테스트 셋 및 샘플 예측 결과
├── recommender.py            # GC‑MC 학습 및 예측 스크립트(메인)
├── requirements.txt          # 의존성 목록
├── 1706.02263v2.pdf          # 관련 논문 PDF (GC‑MC)
└── README.md
```

## 빠른 시작

### 1) 환경 준비

```bash
# (선택) 가상환경 권장
python -m venv .venv && source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2) 학습 및 예측 실행

```bash
# 사용법: python recommender.py <base_file> <test_file>
python recommender.py u1.base u1.test
```

-   스크립트는 내부적으로 `dataset/` 경로를 자동으로 붙여 파일을 로드합니다.
-   예측 결과는 `<base_file>_prediction.txt` 형식으로 프로젝트 루트에 저장됩니다. 예: `u1.base_prediction.txt`

> [!TIP] > `test/` 폴더에 예시 예측 파일(`u1.base_prediction.txt` 등)이 포함되어 있어 출력 포맷을 쉽게 참고할 수 있습니다.

## 데이터

-   `dataset/u.data`: 사용자‑아이템‑평점 탭 구분 파일
-   `dataset/u.item`: 아이템 메타(장르 포함). 인코딩은 ISO‑8859‑1로 로드합니다.
-   `dataset/u.user`: 사용자 메타(나이, 성별, 직업 등)

> [!IMPORTANT] > `recommender.py`는 아이템 파일을 ISO‑8859‑1로 읽도록 구현되어 있습니다. OS 기본 인코딩이 달라도 그대로 사용하세요.

## 구현 개요

-   Graph Convolutional Encoder + Bilinear Decoder(GC‑MC)
-   평점 레벨(1~5)을 edge type으로 모델링하여 기대 평점을 계산
-   선택적으로 사용자/아이템 부가 특징 사용(`use_features=True`)
-   학습 안정화를 위한 learning‑rate warm‑up, ReduceLROnPlateau, gradient clipping, early stopping 등 포함

## 주요 스크립트 옵션(코드 내 기본값)

-   임베딩/은닉 차원: `emb_dim=64`, `hidden_dims=[256, 64]`
-   학습: `epochs=2000`, `batch_size=1024`, `learning_rate=1e-3`, `dropout=0.5`
-   스케줄링/정규화: warm‑up, ReduceLROnPlateau, weight decay, gradient clipping
-   디바이스: CUDA 사용 가능 시 자동 GPU(`--cuda` 플래그 없이 자동 감지)

## 재현 팁

-   동일 데이터 분할(`u1.base / u1.test` 등)을 사용하세요.
-   GPU가 없다면 epochs를 줄이거나 `batch_size`를 축소하세요.
-   재현 가능한 결과를 원하면 실행 전 난수시드를 고정하는 것을 고려하세요.

## 참고문헌

-   Van den Berg, Kipf, & Welling. “Graph Convolutional Matrix Completion.” (arXiv:1706.02263). [arXiv 링크](https://arxiv.org/abs/1706.02263)

## 라이선스

MIT License

Copyright (c) 2025 Junghwan Ryu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

# PyTorch NLP Starter

PyTorch로 빠르게 시작할 수 있는 한국어 감성분류 예제 프로젝트입니다.

## 구조

- `data/sample_sentiment_ko.csv`: 샘플 데이터
- `src/nlp_project/data.py`: 토크나이징, vocab, dataset
- `src/nlp_project/model.py`: BiLSTM 분류기
- `src/nlp_project/train.py`: 학습 엔트리포인트
- `src/nlp_project/infer.py`: 단건 추론 엔트리포인트

## 설치

```bash
pip install -r requirements.txt
```

## 학습

```bash
python -m src.nlp_project.train --epochs 20
```

## 추론

```bash
python -m src.nlp_project.infer --text "정말 재미있고 감동적인 영화였어"
```

## 다음 확장 추천

1. 형태소 분석기(예: mecab) 연동으로 토크나이징 개선
2. 사전학습 모델(KoBERT, KoELECTRA) 파인튜닝으로 성능 향상
3. MLflow/W&B로 실험 추적 추가

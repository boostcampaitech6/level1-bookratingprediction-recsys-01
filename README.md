# Book_Rating_Prediction
## ⭐️ 프로젝트 주제
사용자의 책 평점 데이터를 바탕으로 사용자가 어떤 책을 더 선호할지 예측하는 태스크입니다.

해당 경진대회는 이러한 소비자들의 책 구매 결정에 대한 도움을 주기 위한 개인화된 상품 추천 대회입니다.

리더보드는 평점 예측에서 자주 사용되는 지표 중 하나인 RMSE (Root Mean Square Error)를 사용합니다.

## 🤝프로젝트 팀 구성 및 역할

김수진: 데이터 전처리 및 증강 시도, NCF모델 실험(context data를 활용해 학습, 딥러닝 부분 batchnorm, dropout적용 등), k-fold cross validation 구현 및 학습 단계 최적화 

김예찬: 데이터 전처리 및 원라인 코드 작성, DeepCoNN모델 실험, Catboost 모델 고도화 

남궁진호: 데이터 전처리, DeepCoNN모델 실험

정혜윤: 데이터 전처리, DCN, CNN-DCN 모델 실험, 초기 Cross Validation 코드 일부 작성, Grid Search Cross Validation 코드 작성 및 실험, 데이터 증강 시도

조형진: CNN-FM, CNN-WDN, CNN-DCN, Catboost 모델코드 작성, Optuna로 Catboost 하이퍼파라미터 최적화, Ensemble, Frozen CNN 실험

한예본: 데이터 전처리, CNN-FM, CNN-WDN, CNN-DCN(image data와 context data를 동시에 활용하는 모델 개발), K-modes clustering


## 💻 활용 장비 및 재료

ai stage server : V100 GPU

python==3.10

pytorch==1.12.1 

CUDA==11.3

## 🥇 최종 결과
![최종](https://github.com/boostcampaitech6/level1-bookratingprediction-recsys-01/assets/153365755/eefbf16e-8cdb-4c06-a1f7-bed51f624dc1)

최종순위: 리더보드 최종 Rmse 2.1201 2등

최종모델: CatBoost1(40%), CatBoost2(20%), CNN FM(10%), CNN DCN1(10%), CNN DCN2(20%)

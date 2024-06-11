# 의료보험료 예측을 위한 머신러닝 모델 설계 프로젝트

본 프로젝트는 Kaggle Competition에서 사용된 데이터셋을 이용하여 진행되었다.
 + [의료 보험료 예측 데이터셋](https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction)

=> "나이, 당뇨병 여부, 혈압문제유무, 장기이식여부, 만성질환여부, 키, 몸무게, 알러지유무, 가족 내 암환자 여부, 큰수술 횟수" 의 10가지 특성을 통해 Insurance Price 예측

### 목차

+ [I. 프로젝트 개요](https://github.com/jijeongwon/AI_project/blob/main/README.md#i-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EA%B0%9C%EC%9A%94)
+ II. 데이터 특징공학
+ III. 모델 설명
+ V. 실험 결과
+ IV. 추후 개선 사항

## I. 프로젝트 개요

   #### 1. 프로젝트 배경 및 목적

 최근 의료 보험료 증가로 인해 개인의 의료 보험료 부담이 커지고 있다. 이를 줄이기 위해 의료 보험료를 미리 예측하고 이에 준비하는 것이 중요해졌다.

 본 프로젝트는 머신러닝을 활용하여 개인의 의료 보험료를 예측하는 모델을 설계하는 것을 목표로 한다. 이를 통해 개인이 의료 보험료 지출을 계획하고, 나아가 자신의 높은 보험료 예측값을 보고 경각심을 느껴 건강 관리까지 할 수 있도록 돕고자 하는 것이 목적이다.

   #### 2. 프로젝트의 전반적 설명

 먼저 데이터를 수집하고, 전처리 과정을 거쳐 필요한 특징을 추출한다. 다음으로, 다양한 머신러닝 기법을 적용하여 최적의 예측 모델을 구축한다. 마지막으로 모델의 성능을 평가하고, 예측 정확도를 높이기 위한 방법들을 탐구한다. 최종적으로 구축된 모델을 통해 개인은 자신의 의료 보험료를 예측하고, 지출 계획을 세울 때 예상 보험료에 대한 정보를 얻을 수 있다.

   #### 3. 데이터셋 소개

**Train data 개수 : 788, Test data 개수 : 198**
+ Age : 나이
+ Diabetes : 당뇨병 여부
+ Blood Pressure Problems : 혈압 문제 유무
+ Any Transplants : 장기이식 유무
+ Any Chrnoic Diseases : 만성질환 여부
+ Height : 키
+ Weight : 몸무게
+ Known Allergies : 알러지 유무
+ History of Cancer in Family : 가족 내 암환자 여부
+ Number of Major surgeries : 큰 수술 횟수
  
#### 4. 필요 라이브러리 및 프로그램

+ Python 3, NumPy 1.23, Pandas 1.4, Scikit-learn 1.1, Matplotlib 3.6, Seaborn 0.12
+ Jupyter Notebook

## II. 데이터 특징공학

   #### 1. 기존 데이터셋을 살펴봤을 때 답변 모두 이진화가 완료되어 있었기 때문에 따로 진행하지 않았지만, 기존보다 더 나은 결과를 얻기 위해 Data Synthesis를 진행하여 feature 수를 늘렸다.
   
    # 상관관계 높은 것끼리 데이터 합성

df['synthesis_1'] = df['AnyTransplants'] * df['AnyChronicDiseases']
df['synthesis_2'] = df['AnyTransplants'] * df['NumberOfMajorSurgeries']
df['synthesis_3'] = df['NumberOfMajorSurgeries'] * df['AnyChronicDiseases']
df['synthesis_4'] = df['AnyTransplants'] + df['AnyChronicDiseases'] + df['NumberOfMajorSurgeries']
df['synthesis_5'] = df['BloodPressureProblems'] + df['NumberOfMajorSurgeries']


***

별 세 개를 입력하면 구분선이 생기네요


+ 플러스 표시를 치면 얘가 쩜으로 바뀐대요

+ 지금은 사진 경로를 표시해 볼게요

[이미지](사진.jpg)

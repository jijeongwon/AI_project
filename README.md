***

# 의료보험료 예측을 위한 머신러닝 모델 설계 프로젝트

본 프로젝트는 Kaggle Competition에서 사용된 데이터셋을 이용하여 진행되었다.
 + [의료 보험료 예측 데이터셋](https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction) -> 출처

=> "나이, 당뇨병 여부, 혈압문제유무, 장기이식여부, 만성질환여부, 키, 몸무게, 알러지유무, 가족 내 암환자 여부, 큰수술 횟수" 의 10가지 특성을 통해 Insurance Price 예측

### 목차

+ [I. 프로젝트 개요](https://github.com/jijeongwon/AI_project/blob/main/README.md#i-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EA%B0%9C%EC%9A%94)
+ [II. 데이터 특징 공학](https://github.com/jijeongwon/AI_project/blob/main/README.md#ii-%EB%8D%B0%EC%9D%B4%ED%84%B0-%ED%8A%B9%EC%A7%95-%EA%B3%B5%ED%95%99)
+ [III. 모델 설명](https://github.com/jijeongwon/AI_project/blob/main/README.md#iii-%EB%AA%A8%EB%8D%B8-%EC%84%A4%EB%AA%85)
+ [IV. 실험 결과](https://github.com/jijeongwon/AI_project/blob/main/README.md#iv-%EC%8B%A4%ED%97%98-%EA%B2%B0%EA%B3%BC)
+ [V. Flask 구축](https://github.com/jijeongwon/AI_project/blob/main/README.md#v-flask-%EA%B5%AC%EC%B6%95)
+ [VI. 추후 개선 사항](https://github.com/jijeongwon/AI_project#vi-%EC%B6%94%ED%9B%84-%EA%B0%9C%EC%84%A0-%EC%82%AC%ED%95%AD-%ED%95%9C%EA%B3%84%EC%A0%90)

***

## I. 프로젝트 개요

   #### 1. 프로젝트 배경 및 목적

 최근 의료 보험료 증가로 인해 개인의 의료 보험료 부담이 커지고 있다. 이를 줄이기 위해 의료 보험료를 미리 예측하고 이에 준비하는 것이 중요해졌다.

 본 프로젝트는 머신러닝을 활용하여 개인의 의료 보험료를 예측하는 모델을 설계하는 것을 목표로 한다. 이를 통해 개인이 의료 보험료 지출을 계획하고, 나아가 자신의 높은 보험료 예측값을 보고 경각심을 느껴 건강 관리까지 할 수 있도록 돕고자 하는 것이 목적이다.

   #### 2. 프로젝트 진행 과정

 먼저 데이터를 수집하고, 전처리 과정을 거쳐 필요한 특징을 추출한다. 다음으로, 다양한 머신러닝 기법을 적용하여 최적의 예측 모델을 구축한다. 마지막으로 모델의 성능을 평가하고, 예측 정확도를 높이기 위한 방법들을 탐구한다. 최종적으로 구축된 모델을 통해 개인은 자신의 의료 보험료를 예측하고, 지출 계획을 세울 때 예상 보험료에 대한 정보를 얻을 수 있다.

   #### 3. 데이터셋 소개

Train data 개수 : 788, Test data 개수 : 198 로 나누어주었다.
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

<img width="60%" src="https://github.com/jijeongwon/AI_project/assets/144203449/39d6b291-8c93-41fe-a58f-a7acf03db33c"/>
<img width="60%" src="https://github.com/jijeongwon/AI_project/assets/144203449/2edbc2ad-704b-438a-8e33-0e215719f508"/>
  
#### 4. 필요 라이브러리 및 프로그램

+ Python 3, NumPy 1.23, Pandas 1.4, Scikit-learn 1.1, Matplotlib 3.6, Seaborn 0.12
+ Jupyter Notebook

***

## II. 데이터 특징 공학

   #### 1. 원본 데이터셋을 살펴봤을 때 답변 모두 이진화가 완료되어 있었기 때문에 따로 진행하지 않았지만, 기존보다 더 나은 결과를 얻기 위해 Data Synthesis를 진행하여 feature 수를 늘렸다.

<img width="65%" src="https://github.com/jijeongwon/AI_project/assets/144203449/78ae31fe-12ba-40f7-a0e6-835bff7766cb"/>
   
    # 상관관계 높은 것끼리 데이터 합성

    df['synthesis_1'] = df['AnyTransplants'] * df['AnyChronicDiseases']
    df['synthesis_2'] = df['AnyTransplants'] * df['NumberOfMajorSurgeries']
    df['synthesis_3'] = df['NumberOfMajorSurgeries'] * df['AnyChronicDiseases']
    df['synthesis_4'] = df['AnyTransplants'] + df['AnyChronicDiseases'] + df['NumberOfMajorSurgeries']
    df['synthesis_5'] = df['BloodPressureProblems'] + df['NumberOfMajorSurgeries']

***

## III. 모델 설명

1. 머신러닝 모델 : 다양한 모델링 기법을 사용하며 성능을 개선하기 위해 비교해 보았다. 다음은 여러 모델(LR, DT, MLP, Ada, ...) 중 성능이 가장 높았던 세 가지를 순서대로 나열한 것이다.

(평가 지표는 R-Squared를 사용하였고, Loss는 MAE를 사용하였다.)

+ **Gradient Boosting**
  + learning_rate, n_estimators, max_depth, min_samples_leaf 등의 하이퍼파라미터를 조정해가며 최적의 성능을 도출.
  + **Train : 0.9192, Test : 0.8806**
  + **Loss : 1387.4**

+ **LightGBM**
  + num_leaves, min_child_samples, learning_rate, n_estimators, max_depth 등의 하이퍼파라미터를 조정해가며 최적의 성능을 도출.
  + **Train : 0.9253, Test : 0.8905**
  + **Loss : 1192.5**

+ **Random Forest**
  + n_estimators, max_depth, min_samples_leaf, min_samples_split 등의 하이퍼파라미터를 조정해가며 최적의 성능을 도출.
  + **Train : 0.9152, Test : 0.9020**
  + **Loss : 957.7**

#### 테스트 해본 여러가지 모델 중 Random Forest 모델이 가장 높은 정확도를 가지고 있으며, 다양한 데이터에 대한 처리 및 해석에 유용하다고 생각했기 때문에 최종적으로 이 모댈을 활용하여 의료 보험료 예측 모델을 설계하였다. 

***

성능이 높은 세 가지 모델 이외의 것들

    -----------------------------------------------
    Training Model LR
    Training R-squared: 0.6337588161308307
    Testing R-squared: 0.7218350405543559
    Mean Absolute Error: 2547.1153376972325
    -----------------------------------------------
    Training Model DT 
    Training R-squared: 0.7676052395194115
    Testing R-squared: 0.8793249474437426
    Mean Absolute Error: 1357.778804266782
    -----------------------------------------------
    Training Model MLP Regressor
    Training R-squared: 0.6556950762062116
    Testing R-squared: 0.7407113432497212
    Mean Absolute Error: 2349.4856341313366
    -----------------------------------------------
    Training Model AdaBoost
    Training R-squared: 0.7081650991544655
    Testing R-squared: 0.7983571856679336
    Mean Absolute Error: 2082.1281143876554
    -----------------------------------------------

***

## IV. 실험 결과

+ **다음은 Random Forest 모델을 사용하여 Ablation Study를 진행한 표이다.** 

<img width="60%" src="https://github.com/jijeongwon/AI_project/assets/144203449/af30cf35-48dc-4fde-ab67-81031d9372eb"/>

[III. 모델 설명](https://github.com/jijeongwon/AI_project/blob/main/README.md#iii-%EB%AA%A8%EB%8D%B8-%EC%84%A4%EB%AA%85) 에서도 언급했듯이, 평가 지표는 **R-Squared**를 사용하였고, Loss는 **MAE**를 사용하였다.

표를 보면, 실험 3에서 가장 높은 성능을 얻은 것을 알 수 있다. 하이퍼파라미터는 각각 max_depth=15, n_estimators=50, min_samples_leaf=2, min_samples_split=4로 조정했다.

+ **아래의 그림은 Actual values와 Predicted values의 위치를 비교한 그래프이다.**

<img width="60%" src="https://github.com/jijeongwon/AI_project/assets/144203449/f488bde7-68cd-4192-8154-f8a4c9dade49"/>

그래프를 통해 Train 데이터와 Test 데이터의 Predicted Value를 눈으로 확인할 수 있고, 이들이 Actual Value와는 어느정도 차이가 나는지도 확인이 가능하다.

또한, 이를 통해 모델의 예측 성능을 짐작할 수 있다. 현재 그래프를 보면, Predicted value와 Actual value의 차이가 많이 나지 않고 y축이 0인 지점에 몰려있기 때문에 상당히 예측 성능이 좋은 것을 알 수 있다.






***

## V. Flask 구축

+ 모델링을 마친 후, 마지막으로 Flask 웹 페이지를 구축해보았다. 직접 설계한 머신러닝 모델을 연동한 웹페이지를 만들어 그곳에 자신의 건강 정보를 입력하면, 별도의 코드를 짜는 것 없이 바로 웹페이지에서 자신의 의료보험료를 확인할 수 있게 된다.

(Flask 구축 위해 필요한 코드 모두 업로드 했습니다. -> app.py, home.html, result_high.html, result_mid.html, result_low.html) 





***

## VI. 추후 개선 사항 (+한계점)

1. 데이터의 갯수가 약 1000개로 적은 양이기 때문에 Overfitting은 일어나지 않았지만, 더욱 심도있는 데이터 분석을 하기에는 한계가 있다.

2. Price의 화폐가 무엇인지 알 방법이 없기 때문에 실제 자신의 보험료와 들어맞는지 직접적인 비교는 불가능하고, 자신과 비슷한 건강 정보를 가진 사람의 보험료가 얼마인지만 대략적으로 알 수 있다는 한계가 있다.

3. 모델의 성능을 더욱 올릴 수 있도록 Data Synthesis 외에도 또다른 다양한 feature engineering을 시도해보면 더 좋을 것 같다. (Data Synthesis 후 모델 성능 0.01 올라감.)

***

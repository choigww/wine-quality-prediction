# wine-quality-prediction
Predict wine quality (integer, 1-10) using regression models

```python
Problem name: Wine
URL: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
Dataset: Red
Target variable: quality
Problem type: regression
Data format: semicolon-separated table
Missing values: None
```

#### Input variables (based on physicochemical tests):
변수 의미 출처: https://wikidocs.net/44386

feature|meaning|description
-|-|-
fixed acidity|고정산|와인의 산도를 제어
volatile acidity|휘발산|와인의 향과 연관
citric acid|구연산|와인의 신선함을 올림
residual sugar|잔여 당분|와인의 단 맛 올림
chlorides|염화물|와인의 짠 맛의 원인
free sulfur dioxide|황 화합물|와인을 오래 보관하게 함
total sulfur dioxide|황 화합물|와인을 오래 보관하게 함
density|밀도|와인의 무게감을 나타냄
pH|산성도|와인의 신 맛의 정도
sulphates|황 화합물|와인을 오래 보관하게 함
alcohol|알코올|와인의 단 맛과 무게감에 영향

#### Output variable (based on sensory data): 
12. quality (score between 0 and 10), 와인 품질


## 1. Explanatory Data Analysis

### Sample Size by Label
- 전체적으로 정규분포의 형태를 나타내고 있음
- 특정 target(5,6,7)에 샘플이 집중
    - 교차검증 과정에서 각 Label별 샘플이 골고루 나눠지도록 조치 필요
    - 그러나 target(4, 8, 3)의 경우 샘플 숫자가 너무 적어서 제대로 된 학습이 안될 가능성 높음

![](./img/value-counts-by-target.png)

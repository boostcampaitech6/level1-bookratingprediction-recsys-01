import time
import argparse
import pandas as pd
import numpy as np
from src.utils import Logger, Setting, models_load
from src.data import context_data_load, context_data_split, context_data_loader
from src.data import dl_data_load, dl_data_split, dl_data_loader
from src.data import image_data_load, image_data_split, image_data_loader
from src.data import text_data_load, text_data_split, text_data_loader
from src.train import train, test
# from src.data import integrated_data_load, Integrated_Dataset, integrated_data_split, integrated_data_loader
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error

import catboost
from catboost import CatBoostClassifier, Pool, CatBoostRegressor

import os

import optuna
import joblib
import json



parser = argparse.ArgumentParser(description='parser')
arg = parser.add_argument
arg('--data_path', type=str, default='data/')
arg('--trials', type=int, default=2)

args = parser.parse_args()


now = time.localtime()
now_date = time.strftime('%Y%m%d', now)
now_hour = time.strftime('%X', now)
save_time = now_date + '_' + now_hour.replace(':', '')


## DATA
# context만 쓰니까
data = context_data_load(args) # IMAGE X

# split
# kfold = StratifiedKFold(n_splits=3)
# folds = kfold.split(feature, label)
feature = data['train'].drop(columns='rating')
label = data['train']['rating']
x_test = data['test']


X_train, X_val, y_train, y_val = train_test_split(feature, label, test_size = 0.3, shuffle = True,random_state = 42)



## optuna
def objective(trial):

    param = {
        # 1) 하이퍼 파리미터 목록
        'task_type': 'GPU',
        'devices': 'cuda',

        # GPT
        # 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        # 'depth': trial.suggest_int('depth', 4, 10),
        # 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        # 'border_count': trial.suggest_int('border_count', 32, 255),
        # 'iterations': trial.suggest_int('iterations', 500, 5000),
        # 'random_strength': trial.suggest_float('random_strength', 1, 20),
        # 'bagging_temperature': trial.('bagging_temperature', 0.0, 1.0),
        # 'od_wait':trial.suggest('od_wait', 500,2300),
        
        # DACON
        'iterations':trial.suggest_int("iterations", 500, 3000),
        'od_wait':trial.suggest_int('od_wait', 500, 1500),
        'learning_rate' : trial.suggest_float('learning_rate',0.001, 1),
        'random_strength': trial.suggest_float('random_strength',0,30),
        'depth': trial.suggest_int('depth',3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',1,30),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',1,15),
        'bagging_temperature' :trial.suggest_float('bagging_temperature', 0, 5),
        "l2_leaf_reg": trial.suggest_float('l2_leaf_reg', 1, 20),
        'border_count': trial.suggest_int('border_count', 32, 255),


    }

# "learning_rate": trial.suggest_float('learning_rate',0.001,0.01)
# "depth": trial.suggest_int('depth', 4, 10)
# "l2_leaf_reg": trial.suggest_float('l2_leaf_reg', 2, 10)
# "random_strength": trial.suggest_float(0, 10)

        # DACON
        # 'iterations':trial.suggest_int("iterations", 2000, 10000),
        # 'od_wait':trial.suggest('od_wait', 500,2300),
        # 'learning_rate' : trial.suggest_categorical('learning_rate',[0.01, 0.05,1]),
        # 'reg_lambda': trial.suggest_categorical('reg_lambda',[1e-5,1e-3,1e-1,10,100]),
        # 'depth': trial.suggest_categorical('depth',[3,5,10,15]),
        # 'min_data_in_leaf': trial.suggest_categorical('min_data_in_leaf',[5,10,20,30]),
        # 'leaf_estimation_iterations': trial.suggest_categorical('leaf_estimation_iterations',[1,5,10,13,15]),

        #'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
        #'colsample_bylevel':trial.suggest_float('colsample_bylevel', 0.4, 1.0),
        #'subsample': trial.suggest_float('subsample',0,1),
        #'random_strength': trial.suggest_uniform('random_strength',10,50),
        


    try:
        model = CatBoostRegressor(**param)
        model.fit(X_train, y_train, cat_features=list(feature.columns), verbose=True, eval_set=[(X_val, y_val)])
        pred = model.predict(X_val)

        MSE = mean_squared_error(y_val, pred)
    except Exception as e:
        print(f"오류 발생: {e}")  # 오류 발생 시 출력
    
    print(MSE**0.5)
    return MSE**0.5


## 최적화
# optuna result
study = optuna.create_study(direction='minimize', study_name='catboost_regressor', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=args.trials,n_jobs=1)




# 최적 하이퍼파라미터 출력
print(study.best_params)


## 재학습 후 예측
# 모델 재학습
model_tuned = CatBoostRegressor(**study.best_params, task_type='GPU', devices='cuda')
model_tuned.fit(feature, label, cat_features=list(feature.columns), verbose=False)
result = model_tuned.predict(data['test'])

# model save
joblib.dump(model_tuned, 'saved_models/catboost_optuna'+save_time+'.pkl')

# best parameter 저장

os.makedirs(f'log/{save_time}+catboost', exist_ok=True)

with open(f'log/{save_time}+catboost/catboost_best_hyperparameters_{save_time}.json', 'w') as f:
    json.dump(study.best_params, f)


# submission
submission = pd.read_csv('data/sample_submission.csv')
submission['rating'] = result


submission.to_csv('submit/'+save_time+'CatBoost.csv')


# conda install catboost
# conda config --append channels conda-forge
# conda install optuna
# conda install joblib
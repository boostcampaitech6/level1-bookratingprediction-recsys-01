import time
import argparse
import pandas as pd
from src.utils import Logger, Setting, models_load
from src.data import context_data_load, context_data_split, context_data_loader
from src.data import dl_data_load, dl_data_split, dl_data_loader
from src.data import image_data_load, image_data_split, image_data_loader
from src.data import text_data_load, text_data_split, text_data_loader
from src.train import train, test
from src.data import integrated_data_load, Integrated_Dataset, integrated_data_split, integrated_data_loader
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
import catboost
from catboost import CatBoostClassifier, Pool, CatBoostRegressor

parser = argparse.ArgumentParser(description='parser')
arg = parser.add_argument
arg('--data_path', type=str, default='data/')

args = parser.parse_args()


# context만 쓰니까
data = context_data_load(args)

# split
kfold = StratifiedKFold(n_splits=5)
# model instance
model = CatBoostRegressor(task_type='GPU', devices='cuda')


# K-Fold
feature = data['train'].drop(columns='rating')
label = data['train']['rating']
x_test = data['test']


scores = []
for train_index, test_index in kfold.split(feature, label):

    x_train, x_val = feature.loc[train_index], feature.loc[test_index]
    y_train, y_val = label[train_index], label[test_index]

    model.fit(x_train, y_train, cat_features = list(feature.columns), verbose=False)
    pred = model.predict(x_val)
    scores.append(mean_squared_error(y_val, pred)**0.5)


print(scores)

# submission
submission = pd.read_csv('data/sample_submission.csv')

model.fit(feature, label, cat_features=list(feature.columns), verbose=False)
result = model.predict(x_test)
submission['rating'] = result

now = time.localtime()
now_date = time.strftime('%Y%m%d', now)
now_hour = time.strftime('%X', now)
save_time = now_date + '_' + now_hour.replace(':', '')

submission.to_csv('submit/'+save_time+'CatBoost.csv')
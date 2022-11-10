import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score

import xgboost as xgb

from train_helpers import CATEGORICAL, FEATURES, INT64, prep_df

n_splits = 5
output_file = 'model_midterm.bin'

#################
# Train & Predict
#################

def train(df_train, eta=0.1, max_depth=30, min_child_weight=1):
    df_X, df_y = prep_df(df_train)

    xgb_params = {
        'eta': eta,
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,

        'objective': 'binary:logistic',
        'eval_metric': 'auc',

        'nthread': 8,
        'seed': 1,
        'verbosity': 0,
    }

    dtrain = xgb.DMatrix(df_X, label=df_y, feature_names=FEATURES, enable_categorical=True)

    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=200
    )

    return model

def predict(df, model):
    df_X, _ = prep_df(df)
    df_X_dmatrix = xgb.DMatrix(df_X, feature_names=FEATURES, enable_categorical=True)
    return model.predict(df_X_dmatrix)

#################
# KFold
#################

eta = 0.1
max_depth = 30
min_child_weight = 1

print(f'doing validation with eta={eta}, max_dept={max_depth}, min_child_weight={min_child_weight}')

df = pd.read_csv('data/Train-1617360447408-1660719685476.csv')
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

fold = 0
scores = []

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    model = train(
        df_train,
        eta=eta,
        max_depth=max_depth,
        min_child_weight=min_child_weight
    )

    _, y_val = prep_df(df_val)
    y_pred = predict(df_val, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'auc on fold {fold} is {auc}')
    fold = fold + 1

print('validation results:')
print(
    f'''
        eta={eta}, max_dept={max_depth}, min_child_weight={min_child_weight}:
        {np.mean(scores):.3f} +- {np.std(scores):.3f}
    '''
)

# training the final model

print('training the final model')

model = train(
    df_full_train,
    eta=0.1,
    max_depth=30,
    min_child_weight=1
)

y_pred = predict(df_test, model)
_, y_test = prep_df(df_test)
auc = roc_auc_score(y_test, y_pred)

print(f'auc={auc}')

# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((model, xgb.DMatrix), f_out)

print(f'the model is saved to {output_file}')
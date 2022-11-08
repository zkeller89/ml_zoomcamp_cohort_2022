import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score

import xgboost as xgb

n_splits = 5
output_file = 'model_midterm.bin'

FEATURES = [
    'race',
    'gender',
    'age',
    'time_in_hospital',
    'medical_specialty',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'number_diagnoses',
    'X1',
    'X2',
    'X3',
    'X7',
    'X9',
    'X10',
    'X12',
    'X13',
    'X20',
    'change',
    'diabetesMed'
]

CATEGORICAL = [
    'race',
    'gender',
    'age',
    'medical_specialty',
    'X1',
    'X2',
    'X3',
    'X7',
    'X9',
    'X10',
    'X12',
    'X13',
    'X20',
    'change',
    'diabetesMed'
]

###########
# Data prep
###########

def prep_df(df):
    to_drop = [
        'index', # ID Col
        'patient_id', # ID Col
        'encounter_id', # ID COl
        'weight', # missing
        'diag_1', # high cardinality
        'diag_2', # high cardinality
        'diag_3', # high cardinality
        'diag_4', # high cardinality
        'diag_5', # high cardinality
        'X4', # class imbalance
        'X5', # class imbalance
        'X6', # class imbalance
        'X8', # class imbalance
        'X11', # class imbalance
        'X14', # class imbalance
        'X15', # class imbalance
        'X16', # class imbalance
        'X17', # class imbalance
        'X18', # class imbalance
        'X19', # class imbalance
        'X21', # class imbalance
        'X22', # class imbalance
        'X23', # class imbalance
        'X24', # class imbalance
        'X25', # class imbalance
    ]

    df = df.replace('?', np.nan)
    df = df.drop(to_drop, axis=1)
    nrows = len(df.index)

    ## Group Races
    other_races = ['Hispanic', 'Other', 'Asian']

    # low frequency races --> other
    other_races_ix = df.race.isin(other_races)
    df.race = df.race.mask(other_races_ix, 'Other')

    # missing --> Caucasion
    df.race = df.race.mask(df.race.isna(), 'Caucasian')

    # Gender
    df = df.loc[df.gender != 'Unknown/Invalid']

    ## Medical Specialties
    # Missing as category
    df.medical_specialty = df.medical_specialty.mask(df.medical_specialty.isna(), 'Missing')

    # Low frequency
    ms_low_freq = df.medical_specialty.value_counts(ascending=False) / nrows
    ms_low_freq = ms_low_freq.loc[ms_low_freq < 0.05]
    ms_low_freq = list(ms_low_freq.index)
    df.medical_specialty = df.medical_specialty.mask(df.medical_specialty.isin(ms_low_freq), 'Other')

    # number outpatient - recode to 0,1,2+
    df.number_outpatient = df.number_outpatient.mask(df.number_outpatient >= 2, 2)

    # number inpatient - recode to 0,1,2,3+
    df.number_inpatient = df.number_inpatient.mask(df.number_inpatient >= 3, 3)

    # number emergency - recode to 0,1,2+
    df.number_emergency = df.number_emergency.mask(df.number_emergency >= 2, 2)

    # X1 - recode not "None" to "Other
    df.X1 = df.X1.where(df.X1 == 'None', "Other")

    # X3 - recode "Up", "Down" to "Other"
    df.X3 = df.X3.mask(df.X3.isin(['Up', 'Down']), "Other")

    # X7 - recode not "No" to "Other"
    df.X7 = df.X7.where(df.X7 == 'No', "Other")

    # X9 - recode "Up", "Down" to "Other"
    df.X9 = df.X9.mask(df.X9.isin(['Up', 'Down']), "Other")

    # X10 - recode "Up", "Down" to "Other"
    df.X10 = df.X10.mask(df.X10.isin(['Up', 'Down']), "Other")

    # X12 - recode not "No" to "Other"
    df.X12 = df.X12.where(df.X12 == 'No', "Other")

    # X13 - recode not "No" to "Other"
    df.X13 = df.X13.where(df.X13 == 'No', "Other")

    df[CATEGORICAL] = df[CATEGORICAL].apply(lambda x: x.astype('category'))

    if 'readmitted' in df.columns:
        df_y = df.readmitted
        df_X = df.drop('readmitted', axis=1)
    else:
        df_y = None
        df_X = df

    return df_X, df_y

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
    pickle.dump((model, FEATURES, prep_df), f_out)

print(f'the model is saved to {output_file}')
#%%

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def train_set_missing_ages_model(df):
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    known_age = age_df[age_df.Age.notnull()].values
    y = known_age[:, 0]
    X = known_age[:, 1:]

    set_age_regressor = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=1)
    set_age_regressor.fit(X, y)
    return set_age_regressor


def set_missing_ages(df, regressor):
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    unknown_age = age_df[age_df.Age.isnull()].values
    pred = regressor.predict(unknown_age[:, 1::])
    df.loc[(df.Age.isnull()), 'Age'] = pred


def set_cabin_type(df):
    df.loc[df.Cabin.notnull(), 'Cabin'] = 'YES'
    df.loc[df.Cabin.isnull(), 'Cabin'] = 'NO'


def fit_age_fare_scaler(df):
    scaler = StandardScaler()
    scaler.fit(df[['Age', 'Fare']])
    return scaler


def scale_features(df, scaler):
    scaled_values = scaler.transform(df[['Age', 'Fare']])
    df['Age_scaled'] = scaled_values[:, 0]
    df['Fare_scaled'] = scaled_values[:, 1]
    return scaler


def dummy_features(df):
    dummies_cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
    dummies_embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
    dummies_sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
    dummies_pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')

    df = pd.concat([data_train, dummies_cabin, dummies_embarked, dummies_sex, dummies_pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)


print('done')


#%%

data_train = pd.read_csv('./input/train.csv')
data_test = pd.read_csv('./input/test.csv')


set_ages_regressor = train_set_missing_ages_model(data_train)
set_missing_ages(data_train, set_ages_regressor)
set_missing_ages(data_test, set_ages_regressor)


set_cabin_type(data_train)
set_cabin_type(data_test)


scaler = fit_age_fare_scaler(data_train)
scale_features(data_train, scaler)
scale_features(data_test, scaler)


dummy_features(data_train)
dummy_features(data_test)

print('done')


#%%

from sklearn import linear_model
from sklearn.metrics import accuracy_score
import numpy as np

train_df = data_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values
labels_train = train_np[:, 0]
features_train = train_np[:, 1:]

test_df = data_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
features_test = test_df.values

regression = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
regression.fit(features_train, labels_train)
pred = regression.predict(features_test)
result = pd.DataFrame({'PassengerId': data_test.PassengerId.values, 'Survived': pred.astype(np.int32)})
result.to_csv('./submission.csv', index=False)

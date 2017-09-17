import numpy as np
import pandas as pd
import xgboost as xgb
from  sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier

#load data
train_df = pd.read_csv('train.csv', header=0)
test_df = pd.read_csv('test.csv', header=0)




# data transformation -- filling nan value with mean for numeric otherwise fill with frequent item

class DataImputer(TransformerMixin):
    def __init__(self):
        """impute missing values"""

    def fit(self, X, y= None):
        self.fill = pd.Series([
                        X[c].value_counts().index[0]
                        if X[c].dtype == np.dtype('O')
                        else X[c].mean() for c in X],
                        index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

#considering important columns/features 
feature_columns = ['Pclass','Sex','Age','Fare','Parch']

nonnumeric_colum = ['Sex']

class_colum = ['Survived']

#filter specific columns
tr_x = train_df[feature_columns]
tst_x = test_df[feature_columns]

#append all test rows at the end of training set
X = tr_x.append(tst_x) 

#impute missing or nan values using DataImputer
X_imputed = DataImputer().fit_transform(X)


#convert non-numerical value to numeric using labelencoder
le = LabelEncoder()

for clm in nonnumeric_colum:
    X_imputed[clm] = le.fit_transform(X_imputed[clm])

#seperate train and testing set

train_x = X_imputed[0:train_df.shape[0]].as_matrix()
test_x = X_imputed[train_df.shape[0]::].as_matrix()

train_y = train_df[class_colum[0]]


#classifier construction
model = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
# model = RandomForestClassifier()
model = model.fit(train_x,train_y)
pred = model.predict(test_x)


#keggle format submission

output = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':pred})

output.to_csv("submission.csv",index=False)

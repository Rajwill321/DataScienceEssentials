## Example: LogisticRegression Exercise

impot xlrd
import pandas as pd
cc = pd.read_excel("C:\\Data\\default-cc.xls", header=1)
cc.drop("ID",axis=1,inplace=True)
X=cc.drop("default payment next month", axis=1)
Y=cc['default payment next month']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=.7, random_state=0)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)
from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(Y_test, Y_pred))

#Result Output: 0.7812380952380953

from sklearn.metrics import classification_report
target_columns = [ 'Defaulted', 'NotDefaulted']
print(classification_report(Y_test, Y_pred,target_names=target_columns))

"""
Result:
==========
              precision    recall  f1-score   support

   Defaulted       0.78      1.00      0.88     16408
NotDefaulted       0.25      0.00      0.00      4592

    accuracy                           0.78     21000
   macro avg       0.52      0.50      0.44     21000
weighted avg       0.67      0.78      0.69     21000
"""

## Example: KNN Exercise

impot xlrd
import pandas as pd
cc = pd.read_excel("C:\\Data\\default-cc.xls", header=1)
cc.drop("ID",axis=1,inplace=True)
X=cc.drop("default payment next month", axis=1)
Y=cc['default payment next month']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=.5, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
knnmodel = KNeighborsClassifier(4)
knnmodel.fit(X_train, Y_train)
Y_pred = knnmodel.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(Y_test, Y_pred))

#Result Output: 0.7698

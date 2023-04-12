# Example: Creating Model with SVM / SVC Algorithm
import pandas as pd
bc = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv", header=None)
bc.head()
bc.dtypes
col = [ "IP1", "IP2", "IP3", "IP4", "IP5", "IP6", "IP7", "IP8", "IP8", "OP"]
bc.columns=col
bc.head()
X=bc.drop('OP', axis=1)
Y=bc['OP']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=0)
from sklearn.svm import SVC
SVCmodel = SVC()
SVCmodel.fit(X_train, Y_train)
Y_pred = SVCmodel.predict(X_test)
result = pd.DataFrame({"Actual": Y_test, "Predicted": Y_pred})
result
from sklearn.metrics import accuracy_score, confusion_matrix
print(confusion_matrix(Y_test, Y_pred))
print(accuracy_score(Y_test, Y_pred))
# Save the model
import pickle as pk 
with open("C:\\Data\\models\\svc.ser", "wb") as op:
    pk.dump(SVCmodel, op)

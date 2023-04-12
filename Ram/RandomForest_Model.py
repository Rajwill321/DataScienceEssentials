# Example : Creating Model with RandomForest Algorithm
import pandas as pd
bc = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv", header=None)
bc.head()
bc.dtypes
# Converting non-numerical files to Numberic
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in range(0,9):
    bc[i] = le.fit_transform(bc[i])

# Renaming coloum titles with (IP - Input) & (OP - Output)
col = [ "IP1", "IP2", "IP3", "IP4", "IP5", "IP6", "IP7", "IP8", "IP8", "OP"]
bc.columns=col
X=bc.drop('OP', axis=1)
Y=bc['OP']

# Selecting the data-set 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=0)

# Loading RandomForest algorithm 
from sklearn.ensemble import RandomForestClassifier
Randmodel = RandomForestClassifier()
Randmodel.fit(X_train, Y_train)
Y_pred = Randmodel.predict(X_test)
result = pd.DataFrame({"Actual": Y_test, "Predicted": Y_pred})

# Getting the accuracy 
from sklearn.metrics import accuracy_score, confusion_matrix
print(confusion_matrix(Y_test, Y_pred))
print(accuracy_score(Y_test, Y_pred))

# Save the model
import pickle as pk 
with open("C:\\Data\\models\\randomforest.ser", "wb") as op:
    pk.dump(Randmodel, op)

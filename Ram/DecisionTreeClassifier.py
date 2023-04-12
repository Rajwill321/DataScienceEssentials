# Example: Creating Model with DecisionTreeClassifier

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

# Loading DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
params = {'max_depth': [2,4,6,8,10],
         'min_samples_split': [2,3,4],
         'min_samples_leaf': [1,2]}

clf = DecisionTreeClassifier()
gscv = GridSearchCV(estimator=clf,param_grid=params)
gscv.fit(X_train,Y_train)


from sklearn.metrics import accuracy_score
model = gscv.best_estimator_
model.fit(X_train,Y_train)
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

print(f'Train score {accuracy_score(Y_train_pred,Y_train)}')
print(f'Test score {accuracy_score(Y_test_pred,Y_test)}')

# Save the model
import pickle as pk 
with open("C:\\Data\\models\\decision_tree.ser", "wb") as op:
    pk.dump(gscv, op)

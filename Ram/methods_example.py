### Using Methods to have our code much clean way ###

import pandas as pd
import xlrd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

def get_data(filepath, ext, delimiter=","):
    if ext == "csv":
        df = pd.read_csv(filepath, sep=delimiter)
    elif ext == "xls":
        df = pd.read_excel(filepath)
    
    return df
	
def convert_col_num(df):
    le = LabelEncoder()
    column_list = df.select_dtypes(include = "object").columns
    for col in column_list:
        df[col] = le.fit_transform(df[col])   
		
def train_test(df, output_col):
    X = df.drop([output_col], axis=1)
    Y = df[output_col]
    return train_test_split(X,Y, random_state=0)
	
def analysis(y_test, y_pred):
    print(accuracy_score(y_test, y_pred))
	
# Training with IRIS Data Set
df = get_data("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv", "csv")

X_train, X_test, Y_train, Y_test = train_test(df, 'variety')

model = LogisticRegression(random_state=0)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

analysis(Y_test, y_pred)

#Result = 0.9736842105263158

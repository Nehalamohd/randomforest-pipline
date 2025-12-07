from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from joblib import load
from sklearn.metrics import accuracy_score
import pandas as pd
x,y=load_iris(return_X_y=True)
model=load('model.pkl')
pred=model.predict(x)
acc=accuracy_score(pred,y)
print("acura:",acc)
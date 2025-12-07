from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pandas as pd
data=load_iris(as_frame=True)

df=pd.DataFrame(data.data,columns=data.target)
df.to_csv('iris.csv')
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier()
model.fit(x_train,y_train)
pred=model.predict(x_test)
import joblib
joblib.dump(model,'model.pkl')
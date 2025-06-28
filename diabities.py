import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

df=pd.read_csv("C:/Users/JB978/Downloads/diabetes.csv")


print(df.head())
#x = df.drop(columns=['Outcome'])
x=df.drop(columns=['Outcome'])
y=df['Outcome']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
model=RandomForestClassifier()
model.fit(x_train, y_train)
model.predict(x_train)

new=(10,168	,74,	0	,0	,38,	0.537,	34)
newdata=np.asarray(new).reshape(1,-1)
pre=model.predict(newdata)


if pre[0]==1:
    print('has diabities')
else:
    print('no diabities')



















import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')
#supervised
df=pd.read_csv('diabetes.csv')
x=df.drop(columns='Outcome')
y=df['Outcome']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=12)
NB=GaussianNB()
NB.fit(x_train,y_train)
y_pred=NB.predict(x_test)
#evalution
print(accuracy_score(y_test,y_pred))
final=NB.predict([[10,122,58,21,0,17.6,0,612,48]])
if final==1:
    print("have diabetes")
else:
    print('no diabetes')
    
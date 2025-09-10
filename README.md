# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Collect and preprocess data and scale numerical features

2.import labelencoder and fit transform data

3.import train test split and logistic regression to train features

4.import and print the classification report

## Program:
**Developed by: SHARON CLARA A**

**RegisterNumber:  212224040310**
```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis=1)#removes the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") #A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
<img width="1441" height="268" alt="484909138-6ea64fae-8d32-4a8a-8d93-2b9b482f045e" src="https://github.com/user-attachments/assets/35c4edc6-465b-448c-a015-8aa8e3d96292" />

<img width="1272" height="252" alt="484909228-167dfca8-fd1f-49de-9761-8fda602fb273" src="https://github.com/user-attachments/assets/c5386480-d376-4c80-933a-b5190dc3628b" />
<img width="267" height="382" alt="484909305-31c98df2-0691-4c02-92b4-b0f93480c39c" src="https://github.com/user-attachments/assets/4edab313-5468-4a0e-aea4-61423827502b" />
<img width="1182" height="562" alt="484909683-5d976643-ac61-474b-82c1-4d39ec57008b" src="https://github.com/user-attachments/assets/a83553f6-8b7b-45d5-98f7-625189085600" />
<img width="1105" height="563" alt="484909919-7e6ab18a-54e0-440d-8efc-6c4acf15d5ad" src="https://github.com/user-attachments/assets/6a319c81-0b24-4856-b9f5-86907a65da7c" />
<img width="552" height="348" alt="484910030-bdfdedeb-ca98-441f-b501-247ea0dbc28f" src="https://github.com/user-attachments/assets/d571d2f9-a730-49fc-8999-b47e76ad33f5" />
<img width="898" height="87" alt="484910154-1c83c1e7-0322-44c1-b7e7-765842f64d7d" src="https://github.com/user-attachments/assets/dbac8dbf-fa1b-404e-a08e-d58d49480bd1" />
<img width="529" height="81" alt="484910389-717686e3-6d06-4309-89eb-0d1060b342da" src="https://github.com/user-attachments/assets/bf84a7ba-84fc-4266-8e8a-9474a0eca040" />

<img width="705" height="245" alt="484910482-66f143d8-4247-46a3-bb61-ab7baefb825c" src="https://github.com/user-attachments/assets/4feac940-5ba8-476a-942a-7c4a0c6333af" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

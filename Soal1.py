import matplotlib.pyplot as plt 
import numpy as np
import os 
import pandas as pd 

df = pd.read_csv(r'C:\Users\User\Desktop\tugas no3\fertility.csv')
# print(df.columns.values)

df.drop(['Season'],axis= 'columns', inplace=True)
# df.head()
print(df.info())
print(df['High fevers in the last year'].value_counts()) # Check values per features/columns

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

# Label Encoder
df['Childish diseases'] = label.fit_transform(df['Childish diseases']) 
df['Accident or serious trauma'] = label.fit_transform(df['Accident or serious trauma']) 
df['Surgical intervention'] = label.fit_transform(df['Surgical intervention']) 
df['High fevers in the last year'] = label.fit_transform(df['High fevers in the last year']) 
df['Frequency of alcohol consumption'] = label.fit_transform(df['Frequency of alcohol consumption']) 
df['Smoking habit'] = label.fit_transform(df['Smoking habit']) 
df['Diagnosis'] = label.fit_transform(df['Diagnosis']) 

# Create feature x
x = df.drop(['Diagnosis'], axis=1) # Feature

# Create Target
y = df['Diagnosis']

# One hot encoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

coltrans = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1,2,3,4,5,6])] 
    , remainder='passthrough'
)
x = np.array(coltrans.fit_transform(x), dtype=np.float64)
print(x[1])

# Train and Split test train data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size = .1
)
#---------------  ML  -----------------------#
# Logistic regression (modellog)
from sklearn.linear_model import LogisticRegression
modellog = LogisticRegression(solver='liblinear')
modellog.fit(x_train, y_train)
print(round(modellog.score(x_test, y_test)*100,2),'%')

# Decision Trees
from sklearn import tree
modelDT = tree.DecisionTreeClassifier()
modelDT.fit(x_train,y_train)
print(round(modelDT.score(x_test,y_test)*100,2),'%')

# Kneighbors
from sklearn.neighbors import KNeighborsClassifier
modelKn = KNeighborsClassifier()
modelKn.fit(x_train,y_train)
print(round(modelKn.score(x_test,y_test)*100,2),'%')


# --------------------- Prediction ----------------------- #
pred = [[ 1,  0,  1,  0,  0,  0,  1,  1,  0,  0,  1,  1,  0,  0,  1,  0,  0, 29, 5]]
# Logistic regression
if int(modellog.predict(pred)) == 1:
    diagnoselog = 'Normal'
else:
    diagnoselog = 'Altered'
# Decision Trees
if int(modelDT.predict(pred)) == 1:
    diagnoseDT = 'Normal'
else:
    diagnoseDT = 'Altered'
# K nearest neighbour
if int(modelKn.predict(pred)) == 1:
    diagnoseKn = 'Normal'
else:
    diagnoseKn = 'Altered'
    
print('Arin, prediksi kesuburan: ',diagnoselog,' (Logistic Regression)')
print('Arin, prediksi kesuburan: ',diagnoseDT,' (Decision Trees)')
print('Arin, prediksi kesuburan: ',diagnoseKn,' (K-nearest  neigbours)')
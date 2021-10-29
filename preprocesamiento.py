import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

''' PRE-PROCESAMIENTO DE DATOS 
1   Definir tipo de variables
     Country - Nominal
     Age - Discreta
     Salary - Discreta
     Purchased - Nominal     
'''
dataset=pd.read_csv('Data.csv')

# Se obtienen las variables independientes en un arreglo aparte X
X = dataset.iloc[:,:-1].values
# Se obtiene la variable dependiente en un arreglo aparte Y
Y = dataset.iloc[:,-1].values
#Importa el método para imputar los datos
from sklearn.impute import SimpleImputer
#Generar variable para nuevos valores de nan usando la media
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
#Efectuar la imputación sobre las columnas que queremos modificar
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#Codificación disyuntiva - variables dummy
ct = ColumnTransformer(
                        transformers=[('encoder',OneHotEncoder(),[0])],
                        remainder='passthrough')
X = np.array(ct.fit_transform(X))

#Reemplazar por una codificación binaria la variable dependiennte
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

#Dividir conjunto de datos en entrenamiento y pruebas
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_text = train_test_split(
    X,Y,test_size=0.2,random_state=1)

#Estandarización
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,3:] = sc.fit_transform(X_train[:,3:])
X_test[:,3:] = sc.fit_transform(X_test[:,3:])



















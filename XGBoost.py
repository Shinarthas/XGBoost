import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  # все кроме последней колонки
y = dataset.iloc[:, -1].values  # только последняя колонка

# разодьем данные на тестовые и проверочные
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=1)#random_state=1 убирает рандом что он всегжа одинаков

# feature scaling
from sklearn.preprocessing import StandardScaler
ss=StandardScaler() #сколько среднеквадратичных отклонений содержит наша величина
X_train[:,3:]=ss.fit_transform(X_train[:,3:])#применяем к тестовой выборке
# когда мы вызываем fit_transform мы (1) готовим модель кторая конвертирует, а потом на основе ее изменяем наши данные
X_test[:,3:]=ss.transform(X_test[:,3:]) # тут только transform потому что мы ТОЛЬКО ЧТО создали модель странсформации, и среднее и отклонение УЖЕ расчитаны, поэтому только меняем

#Training XGBoost on the training set
from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(X_train, y_train)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('--kernel SVM--')
print(cm)
print(accuracy_score(y_test, y_pred))

#applying k-fold cross validation
from sklearn.model_selection import cross_val_score
#прогонит все через N итераций и тестовая и тренееровочная выборка всегда будут разные, делаем мы это на тренировочной сессии
accuacies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
print("Accurace {:.2f} %".format(accuacies.mean()*100))
print("Standart Deviation {:.2f} %".format(accuacies.std()*100))

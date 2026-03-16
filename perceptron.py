#f(weights*x+b)
#x = input , b = bias
#step activation function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
data = load_breast_cancer()
x = data.data
y = data.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=43)
model = Perceptron(penalty=None,
                   alpha=0.001,
                   fit_intercept=True,
                   max_iter=1000,
                   tol=1e-3,
                   shuffle=True,
                   eta0=1.0,
                   random_state=43,
                   early_stopping=True,
                   class_weight="balanced",
                   warm_start=True)
'''
parameters:
1.penalty
2.alpha
3.fit_intercept
4.max_iter --> max no of epochs(full connection passes through the dataset)
5.tol(tolerance) --> 1e-3
6.eta0 --> learning rate,weights_new = weights_old + learnin_rate*error*x
eta0 = 1.0
7.shuffle'''
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(model.coef_)
print(model.intercept_)
print(model.coef_.shape)
print(model.intercept_.shape)
print(model.n_iter_) #no of epochs the model took for the weights to convergence as positive negative
print(model.t_) #total no of weights updates

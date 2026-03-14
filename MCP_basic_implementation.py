import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
class mcp:
    def __init__(self,weights,threshold):
        self.weights = weights
        self.threshold = threshold
    def predicted(self,x):
        summation = np.dot(x,self.weights)
        return np.where(summation>= self.threshold,1,0)

#creating AND Gate
x = np.array([
    [0,0],[0,1],[1,0],[1,1]
])
y = np.array([0,0,0,1])
weights = [1,1]
threshold = 2
model  = mcp(weights,threshold)
y_pred = model.predicted(x)
print(y_pred)
print(accuracy_score(y,y_pred))
#IT IS THE FIRST ALGORITHM OF NEURAL NETWORKS

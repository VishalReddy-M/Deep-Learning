import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
data = load_breast_cancer()
df = pd.DataFrame(data.data,columns=data.feature_names)
df["target"] = data.target
print(df)
print(df.isna().sum())
print(df["target"].value_counts())
sns.countplot(x="target",data=df)
plt.title("target display")
plt.show()
df["mean radius"].hist()
plt.show()
#model
x = data.data
y = data.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=43)
scale = StandardScaler()
x_train_scaled = scale.fit_transform(x_train)
x_test_scaled = scale.transform(x_test)
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation="sigmoid", input_shape=(x_train_scaled.shape[1],))
])
# tensorflow is a DL library it helps us to build,train,optimize the weights
# it is like stacking layers one by one
#dense(1)--> only one neuron due to single layer perceptron
model.compile(optimizer = "adam",loss = "binary_crossentropy",metrics = ["accuracy"])
# model training
history = model.fit(x_train_scaled,y_train,epochs = 15,batch_size = 30,validation_split = 0.2)
loss,accuracy = model.evaluate(x_test_scaled,y_test)
print(accuracy,loss)
y_pred = model.predict(x_test_scaled)
y_pred_modi = (y_pred>0.5).astype(int)
print(accuracy_score(y_test,y_pred_modi))
print(confusion_matrix(y_test,y_pred_modi))
print(classification_report(y_test,y_pred_modi))
print(history.history.keys())
plt.figure(figsize=(7,7))
plt.plot(history.history["loss"],linewidth = 10,linestyle = "--")
plt.plot(history.history["val_loss"],color = "red")
plt.title("model loss curve")
plt.grid()
plt.legend(["training loss","validation loss "])
plt.show()
plt.figure(figsize=(7,7))
plt.plot(history.history["accuracy"],linewidth = 10,linestyle = "--")
plt.plot(history.history["val_accuracy"],color = "red")
plt.title("model accuracy curve")
plt.grid()
plt.legend(["training accuracy","validation accuracy"])
plt.show()
#gradient descent is used to update weights to minimize the loss error
# optimizer internally calculates the gradient descent and update weights

 
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as pyplot

iris = load_iris()
x, y = iris.data, iris.target

#convert to (dataframe) for better visualization
iris_df = pd.DataFrame(x, columns=iris.feature_names)
print(iris_df.head())
      
# print the targeted names
print("Targeted names: ", iris.target_names)

# disperse the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3 ,random_state=42)

# setup/bring in the K Nearest Neighbors (KNN) classifier
model  = KNeighborsClassifier()
model.fit(x_train, y_train)

# make the prediciton
y_pred = model.predict(x_test)

# evaluate the model
print(confusion_matrix(y_test, y_pred))     # print the confusion matrix
print(classification_report(y_test, y_pred))        # print the classification report

# plot the predicted values and actual values
pyplot.scatter(range(len(y_test)), y_test, color='blue', label='Real')
pyplot.scatter(range(len(y_pred)), y_pred, color='yellow', label='Predicted', alpha=0.5)
pyplot.legend()
pyplot.title('Real vs Predicted')
pyplot.show()
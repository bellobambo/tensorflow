import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("data.csv", sep=";")
data = data[["Marital status", "Application mode", "Application order", "Course"]]

predict = "Marital status"

X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

#
# best = 0
# for _ in range(15):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
#
#     linear = linear_model.LinearRegression()
#
#     linear.fit(x_train, y_train)
#     acc = linear.score(x_test, y_test)
#     print(acc)
#     print("-------")
#
#     if acc > best:
#         best = acc
#         with open("studentmodel.pickle", "wb") as f:
#             pickle.dump(linear, f)


pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p="Course"
style.use("ggplot")
pyplot.scatter(data[p], data["Marital status"])
pyplot.xlabel(p)
pyplot.ylabel("Final Label")
pyplot.show()

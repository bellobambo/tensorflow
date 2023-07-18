import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

data = pd.read_csv("name_gender_dataset.csv")
data = data[["Name", "Gender", "Count", "Probability"]]
print(data.head())

predict = "Name"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

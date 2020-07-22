import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

USAhousing = pd.read_csv('Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression/USA_Housing.csv')

USAhousing.head()

USAhousing.info()

USAhousing.describe()

USAhousing.columns

sns.pairplot(USAhousing)

sns.distplot(USAhousing['Price'])

sns.heatmap(USAhousing.corr())

X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]

y = USAhousing['Price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
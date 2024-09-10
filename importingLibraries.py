import numpy as np 
import pandas as pd 

# Libraries for visualisation
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

# Libraries for Preprocessing
# StandardScaler: It is a tool for standardizing numeric features.Ensuring each feature has a mean 0 and variance 1.
# OneHotEncoder: It convert categorical variables into one hot numeric array.
# OptimalEncoder: Encode categorical feature as integer based on the order
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer

# Machine Learning Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Libraries for Metrics
from sklearn.metrics import accuracy_score, confusion_matrix

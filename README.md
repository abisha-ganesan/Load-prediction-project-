# Load-prediction-project-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
url = "https://raw.githubusercontent.com/dsrscientist/DSData/master/loan_prediction.csv"
data = pd.read_csv(url)
data.head()
data = data.dropna()
data.isnull().sum()  # check again
data['Gender'] = data['Gender'].map({'Male':1, 'Female':0})
data['Married'] = data['Married'].map({'Yes':1, 'No':0})
data['Education'] = data['Education'].map({'Graduate':1, 'Not Graduate':0})
data['Self_Employed'] = data['Self_Employed'].map({'Yes':1, 'No':0})
data['Property_Area'] = data['Property_Area'].map({'Urban':2, 'Semiurban':1, 'Rural':0})
data['Loan_Status'] = data['Loan_Status'].map({'Y':1, 'N':0})
X = data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']]
y = data['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=52)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# This is a simple logistic regression model to predict loan approvals.
# We used features like income, loan amount, and credit history.
# The model achieved around __% accuracy.

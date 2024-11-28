import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier

# loading dataset
data = load_breast_cancer()
y = data["target"]
x = data["data"]

# setting train and test set. Any change in test size and random state can change the results
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# result by using bayes
bayes_decision = GaussianNB()
bayes_decision.fit(x_train, y_train)
# result by using logic regression
logic_regression = make_pipeline(StandardScaler(), LogisticRegression())
logic_regression.fit(x_train, y_train)

print("Bayes decision results:")
g_predictions = bayes_decision.predict(x_test)
print(g_predictions)
print("Logic Regression results: ")
l_predictions = logic_regression.predict(x_test)
print(l_predictions)

g_score = bayes_decision.score(x_test, y_test)
l_score = logic_regression.score(x_test, y_test)
print(g_score)
print(l_score)

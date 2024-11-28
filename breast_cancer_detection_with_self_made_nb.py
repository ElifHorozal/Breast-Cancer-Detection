import math
import numpy as np
import pandas as pd

# necessary functions


def prior_probability(df, ds):
    a = df.shape[0]
    b = ds.shape[0]
    return a / b


def mean(df):
    x = df.mean(0)
    return x


def covariance(df):
    df = np.transpose(df)
    x = np.cov(df)
    return x


def bayes(d_test, df, ds):
    size = df.shape[1]
    cov = covariance(df)
    sub = np.subtract(d_test, mean(df))
    top = math.exp(-0.5 * np.inner(sub, np.inner(sub, np.linalg.inv(cov))))
    bottom = math.pow(2 * math.pi, size / 2) * math.pow(np.linalg.det(cov), 1 / 2)
    x = (top * prior_probability(df, ds)) / bottom
    return x


# imported dataset
dataset = pd.read_csv("wdbc.data")
# first 469 sample are used for train, last 99 sample used for test
train_set = dataset.head(469)
test_check = dataset.tail(99)
del test_check["0"]
test_set = dataset.tail(99)
del test_set["0"]
del test_set["1"]
# split the data of two classes
m_df = train_set.loc[train_set["1"] == "M"]
del m_df["0"]
del m_df["1"]
b_df = train_set.loc[train_set["1"] == "B"]
del b_df["0"]
del b_df["1"]

# turned the data into matrix
train_set = np.matrix(train_set.to_numpy())
test_check = np.matrix(test_check.to_numpy())
test_set = np.matrix(test_set.to_numpy())
m_df = np.matrix(m_df.to_numpy())
b_df = np.matrix(b_df.to_numpy())

# calculating values, comparing them with each other to decide which class they belong and record them
true = 0
count = 0
predictions = np.array(99)
while count < test_set.shape[0]:
    test = test_set[count, :]
    m_bayes = bayes(test, m_df, train_set)
    b_bayes = bayes(test, b_df, train_set)

    if m_bayes > b_bayes:
        y = "M"
        predictions = np.append(predictions, 1)
    else:
        y = "B"
        predictions = np.append(predictions, 0)

    if y == test_check[count, 0]:
        true = true + 1

    arr = [[y, test_check[count, 0]]]
    count = count + 1

# the results as predictions and accuracy of them
print(predictions)
print(true / test_set.shape[0])

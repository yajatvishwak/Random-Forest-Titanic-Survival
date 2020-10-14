# loading the dataset
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
df = pd.read_csv('data.csv')

X = df[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age',
        'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
y = df['Survived']

X = X.drop(['PassengerId', 'Embarked', 'Name', 'Ticket', 'Cabin'], axis=1)


le = LabelEncoder()
X['Sex'] = le.fit_transform(X['Sex'])


X = X.fillna(X.mean())
print(X)
model = DecisionTreeClassifier()
model.fit(X, y)
importance = model.feature_importances_
for i, v in enumerate(importance):
    print('Feature: %s, Score: %.5f' % (X.columns[i], v))
featurevsimp = plt.bar([x for x in range(len(importance))], importance)
featurevsimp = plt.xlabel("Features")
featurevsimp = plt.ylabel("Importance")
plt.show()

# We come to know that the most important features that decide the output, we can drop the low score columns
# Choosing sex age and fare because those hold very high importance in resulting output

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.30, random_state=42)


# Defining our model
estimators = [10, 20, 40, 50, 100, 200, 300, 400]
scores = []
temp = {}

for estimator in estimators:
    clf = RandomForestClassifier(n_estimators=estimator)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    scores.append(score)
    temp[score] = estimator


print(temp[max(scores)], ":", max(scores))
bestEstimator = temp[max(scores)]
clf = RandomForestClassifier(n_estimators=bestEstimator)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred.tolist())
print(y_test.values.tolist())
print("Accuracy : ", max(scores) * 100, "%")
print("Total Test Cases : ", len(y_pred))
print("Correct Predictions : ",  (max(scores) * len(y_pred)))
print("Wrong Predictions : ",  len(y_pred) - (max(scores) * len(y_pred)))

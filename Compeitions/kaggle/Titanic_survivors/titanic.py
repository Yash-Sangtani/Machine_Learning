import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')
y = train_data['Survived']
passenger_id = test_data['PassengerId']
train_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
avg_age = train_data.loc[:, 'Age'].mean()
avg_fare = train_data.loc[:, 'Fare'].mean()
train_data['Age'].fillna(avg_age, inplace=True)
avg_age = test_data.loc[:, 'Age'].mean()
avg_fare = test_data.loc[:, 'Fare'].mean()
test_data['Age'].fillna(avg_age, inplace=True)
test_data['Fare'].fillna(avg_fare, inplace=True)
lr.fit(train_data, y)
print(lr.score(train_data, y))
print(test_data.columns)
y_output = lr.predict(test_data)
print(len(y_output))
gender_submission = pd.DataFrame(
        {
            'PassengerId': passenger_id,
            'Survived': y_output
        })
print(gender_submission)
gender_submission.to_csv('gender_submission.csv', index=False)

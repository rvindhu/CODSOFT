import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\rvind\\Downloads\\Titanic-Dataset.csv")
print(data)
print(data.info())
print(data.describe())
print(data.describe().round(1))

print(data.isna().sum())
sns.heatmap(data.isna(), cbar=False)
plt.title('Missing Values Heatmap')
plt.show()

data['Age'] = data['Age'].fillna(data['Age'].median())
sns.heatmap(data.isna(), cbar=False)
plt.title('Missing Values Heatmap After Filling Age')
plt.show()

data.drop('Cabin', axis=1, inplace=True)
sns.heatmap(data.isna(), cbar=False)
plt.title('Missing Values Heatmap After Dropping Cabin')
plt.show()

print(data.Embarked.value_counts().reset_index())
sns.heatmap(data.isna(), cbar=False)
plt.title('Missing Values Heatmap After Embarked Analysis')
plt.show()

data['Sex'] = pd.get_dummies(data['Sex'], drop_first=True, dtype=int)
sex = pd.get_dummies(data['Sex'], dtype=int)
emp = pd.get_dummies(data['Embarked'], dtype=int)

data = pd.concat([data, emp], axis=1)
data.drop(['PassengerId', 'Name', 'Ticket', 'Embarked'], axis=1, inplace=True)
print(data)

sns.heatmap(data.corr(), annot=True)
plt.title('Correlation Heatmap')
plt.show()

sns.countplot(x='Survived', data=data)
plt.title('Survived Count Plot')
plt.show()

print(data.Survived.value_counts().reset_index())
print(data.Fare.unique())
print(data.Fare.value_counts().reset_index())

g = data.Fare.groupby(data.Survived).value_counts().reset_index()
print(g)
print(g[g.Survived == 1])

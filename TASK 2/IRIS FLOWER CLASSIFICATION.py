import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import scikitplot as skplt
import numpy as np

# Load and inspect the dataset
df = pd.read_csv("C:\\Users\\rvind\\Downloads\\IRIS.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Encode categorical labels
df["species"] = df["species"].replace({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
label_name = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

# Visualizations
sns.countplot(y="species", data=df)
plt.title('Species Count')
plt.show()

sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="species")
plt.title('Sepal Length vs Sepal Width')
plt.show()

sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="species")
plt.title('Petal Length vs Petal Width')
plt.show()

sns.histplot(data=df, x="sepal_length", color="red", kde=True)
plt.title('Sepal Length Distribution')
plt.show()

sns.histplot(data=df, x="sepal_width", color="navy", kde=True)
plt.title('Sepal Width Distribution')
plt.show()

sns.histplot(data=df, x="petal_length", color="darkgreen", kde=True)
plt.title('Petal Length Distribution')
plt.show()

sns.histplot(data=df, x="petal_width", color="darkorange", kde=True)
plt.title('Petal Width Distribution')
plt.show()

sns.pairplot(df, hue="species")
plt.title('Pairplot of Features')
plt.show()

# Data preparation
X = df.drop(columns="species")
y = df["species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
logistic_model = LogisticRegression()
svm_model = SVC(probability=True)
random_forest_model = RandomForestClassifier()

# Train and evaluate Logistic Regression
logistic_model.fit(X_train, y_train)
pred1 = logistic_model.predict(X_test)

print(f"Logistic Regression Accuracy Score: {accuracy_score(y_test, pred1)}")

cf = confusion_matrix(y_test, pred1)
sns.heatmap(cf, annot=True, fmt="d", cmap="cividis", xticklabels=label_name, yticklabels=label_name)
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

print(classification_report(y_test, pred1, target_names=label_name))

# ROC AUC for Logistic Regression
y_probas_logistic = logistic_model.predict_proba(X_test)
roc_auc_logistic = roc_auc_score(y_test, y_probas_logistic, multi_class='ovr')
print("Logistic Regression ROC AUC Score:", roc_auc_logistic)

skplt.metrics.plot_roc(y_test, y_probas_logistic)
plt.title('Logistic Regression ROC Curve')
plt.show()

# Train and evaluate SVM
svm_model.fit(X_train, y_train)
pred2 = svm_model.predict(X_test)

print(f"SVM Accuracy Score: {accuracy_score(y_test, pred2)}")

cf = confusion_matrix(y_test, pred2)
sns.heatmap(cf, annot=True, fmt="d", cmap="hot", xticklabels=label_name, yticklabels=label_name)
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

print(classification_report(y_test, pred2, target_names=label_name))

# ROC AUC for SVM
y_probas_svm = svm_model.predict_proba(X_test)
roc_auc_svm = roc_auc_score(y_test, y_probas_svm, multi_class='ovr')
print("SVM ROC AUC Score:", roc_auc_svm)

skplt.metrics.plot_roc(y_test, y_probas_svm)
plt.title('SVM ROC Curve')
plt.show()

# Cross-validation for SVM
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
cross_val_results_svm = cross_val_score(svm_model, X, y, cv=kf)
print(f'SVM Cross-Validation Results (Accuracy): {cross_val_results_svm}')
print(f'SVM Mean Accuracy: {cross_val_results_svm.mean()}')

# Train and evaluate Random Forest
random_forest_model.fit(X_train, y_train)
pred3 = random_forest_model.predict(X_test)

print(f"Random Forest Accuracy Score: {accuracy_score(y_test, pred3)}")

cf = confusion_matrix(y_test, pred3)
sns.heatmap(cf, annot=True, fmt="d", cmap="spring", xticklabels=label_name, yticklabels=label_name)
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

print(classification_report(y_test, pred3, target_names=label_name))

# ROC AUC for Random Forest
y_probas_rf = random_forest_model.predict_proba(X_test)
roc_auc_rf = roc_auc_score(y_test, y_probas_rf, multi_class='ovr')
print("Random Forest ROC AUC Score:", roc_auc_rf)

skplt.metrics.plot_roc(y_test, y_probas_rf)
plt.title('Random Forest ROC Curve')
plt.show()

# Cross-validation for Random Forest
cross_val_results_rf = cross_val_score(random_forest_model, X, y, cv=kf)
print(f'Random Forest Cross-Validation Results (Accuracy): {cross_val_results_rf}')
print(f'Random Forest Mean Accuracy: {cross_val_results_rf.mean()}')

# Custom data prediction
test_df = pd.DataFrame([[5.1, 3.5, 1.4, 0.3]], columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
print("Random Forest Prediction for custom data:", random_forest_model.predict(test_df))

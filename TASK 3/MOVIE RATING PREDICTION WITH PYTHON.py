import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.offline import iplot
from plotly.subplots import make_subplots
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import RidgeCV
import sklearn.metrics as metrics
import scikitplot as skplt
import category_encoders as ce

warnings.simplefilter(action="ignore")
sns.set_theme(palette=sns.color_palette("muted"), style="darkgrid")

with open(r"C:\\Users\\rvind\\Documents\\IMDb Movies India.csv", 'r', encoding='latin1') as f:
    data = pd.read_csv(f)

data = data.dropna(subset=["Year", "Genre", "Director", "Actor 1", "Actor 2", "Actor 3", "Rating"])
data = data.dropna().reset_index(drop=True)


data["Duration"] = data["Duration"].str.replace(" min", "").astype(float)
data["Votes"] = data["Votes"].str.replace(",", "").astype(float)


data = data.drop_duplicates(keep="first", subset=["Name", "Year"]).reset_index(drop=True)

# Cleaning Genre and year columns
data["Year"] = data["Year"].str.extract(r"(\d{4})").astype(float)
data["Genre"] = data["Genre"].str.replace("Musical", "Music")
data["Main_genre"] = data["Genre"].str.extract("(^\w{1,11})")

# Removing outliers using z-score method
data = data[(np.abs(stats.zscore(data[['Rating', 'Votes', 'Duration']])) < 3).all(axis=1)]

# Univariate Analysis
iplot(px.violin(data_frame=data, x="Rating", title="Distribution of Ratings"))
iplot(px.violin(data_frame=data, x="Duration", title="Distribution of Duration"))
iplot(px.violin(data_frame=data, x="Votes", title="Distribution of Votes"))

# Distribution of movies across Genres
genres = data['Main_genre'].value_counts().reset_index()
genres.columns = ['Main_genre', 'count']  # Rename columns for clarity
iplot(px.pie(data_frame=genres, names='Main_genre', values='count', title="Number of Movies by Genre", height=1050).update_traces(textinfo="value+percent"))

# Multivariate Analysis
rating_by_genre = data.groupby("Main_genre")["Rating"].mean().sort_values(ascending=False)
iplot(px.bar(data_frame=rating_by_genre.reset_index(), x="Main_genre", y="Rating", title="Average Rating by Genre"))

# Distribution of movies over time
movies_by_year = data["Year"].value_counts().reset_index().sort_values(by="Year")
movies_by_year.columns = ['Year', 'count']  # Rename columns for clarity
iplot(px.line(data_frame=movies_by_year, x="Year", y="count", title="Number of Movies over the Years", color_discrete_sequence=["green"]))

# Rating and number of votes over the years
Rating_by_years = data.groupby("Year").agg({"Rating": "mean", "Votes": "sum"}).reset_index()
iplot(px.line(data_frame=Rating_by_years, x="Year", y="Rating", markers=True, color_discrete_sequence=["green"], height=400))
iplot(px.line(data_frame=Rating_by_years, x="Year", y="Votes", color_discrete_sequence=["Red"], markers=True, height=400))

# Top 10 by Rating
def top_10_rating(col):
    return data.groupby(col)["Rating"].agg(["mean", "count"])\
    .query("count >=10")\
    .sort_values(by="mean", ascending=False)[:10]\
    .reset_index()

top_10_director = top_10_rating("Director")
top_10_actors1 = top_10_rating("Actor 1")
top_10_actors2 = top_10_rating("Actor 2")
top_10_actors3 = top_10_rating("Actor 3")

iplot(px.bar(data_frame=top_10_director, x="Director", y="mean", text="count", labels={'mean': 'Rating', 'count': 'Number of movies'}, title="Top 10 Directors by Rating"))
iplot(px.bar(data_frame=top_10_actors1, x="Actor 1", y="mean", text="count", labels={'Actor 1': 'Main Actor', 'mean': 'Rating', 'count': 'Number of movies'}, title="Top 10 Main Actors by Rating"))
iplot(px.bar(data_frame=top_10_actors2, x="Actor 2", y="mean", text="count", labels={'mean': 'Rating', 'count': 'Number of movies'}, title="Top 10 Secondary Actors by Rating"))
iplot(px.bar(data_frame=top_10_actors3, x="Actor 3", y="mean", text="count", labels={'mean': 'Rating', 'count': 'Number of movies'}, title="Top 10 Third Actors by Rating"))

# Machine Learning Model
def regression_results(y_true, y_pred):
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance, 4))
    print('mean_squared_log_error: ', round(mean_squared_log_error, 4))
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mean_absolute_error, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse), 4))
    print('Median absolute error: ', round(median_absolute_error, 4))

# Dropping unnecessary columns and splitting data into X and y
data = data.drop(columns=["Name", "Main_genre"])
X = data.drop(columns="Rating")
y = data["Rating"]

# Encoding data
encoder = ce.JamesSteinEncoder(return_df=True)
encoder.fit(X, y)
X = encoder.transform(X)

# Normalizing data
scaler = RobustScaler()
scaler.fit(X)
X = scaler.transform(X)

# Training XGBoost model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, shuffle=True)
xgb_model = XGBRegressor(objective='reg:squarederror', gamma=0.09, learning_rate=0.08, subsample=0.7)
xgb_model.fit(X_train, y_train)

# Evaluating XGBoost model
print("XGBRegressor")
print(f"Training score: {xgb_model.score(X_train, y_train)}")
print(f"Testing score: {xgb_model.score(X_test, y_test)}")
y_pred = xgb_model.predict(X_test)
print("Report: XGBoost model")
regression_results(y_test, y_pred)

# Cross-validation
score = cross_val_score(xgb_model, X, y, cv=10)
avg = np.mean(score)
print(f"Cross-validation score for XGBoost: {score}")
print(f"Average cross-validation score for XGBoost: {avg}")

# Feature Importances
fs = xgb_model.feature_importances_
feature_names = [f'Feature {i}' for i in range(len(fs))]  # Create generic feature names
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': fs}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(12, 9))
plt.title("Feature Importances")
plt.bar(x=feature_importances['Feature'], height=feature_importances['Importance'])
plt.xticks(rotation=90)
plt.show()

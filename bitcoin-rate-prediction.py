
#Data Source
#https://www.cryptodatadownload.com/data/bitfinex/


import zipfile

with zipfile.ZipFile("/content/BitFinexData.zip", "r") as dataset:
  dataset.extractall("/content/bitfinex/")

import pandas as pd
bitcoin = pd.read_csv("/content/bitfinex/Bitfinex_BTCUSD_d.csv", skiprows=1)

bitcoin.head()

bitcoin.describe()

bitcoin.open.plot()

bitcoin["Volume BTC"].plot()

bitcoin["Volume USD"].plot()

bitcoin.head()


bitcoin["open_mean_14d"] = bitcoin["open"][::-1].rolling(window=14).mean() # Средняя цена открытия за последние 14 дней
bitcoin["close_max_7d"] = bitcoin["close"][::-1].rolling(window=7).max()

for day in range(1,8):
  bitcoin[f"close_day_{day}"] = bitcoin["close"][::-1].shift(day+1)

bitcoin["dt"] = pd.to_datetime(bitcoin["date"])

bitcoin["weekday"] = bitcoin["dt"].dt.weekday
bitcoin["month"] = bitcoin["dt"].dt.month
bitcoin["year"] = bitcoin["dt"].dt.year

bitcoin["target"] = bitcoin["close"].shift(1)

bitcoin[["date", "close", "target"]].head()

bitcoin = pd.get_dummies(bitcoin, columns=["weekday", "month", "year"])
bitcoin.drop("date", axis=1, inplace=True)
bitcoin.drop("unix", axis=1, inplace=True)
bitcoin.drop("symbol", axis=1, inplace=True)
bitcoin.drop("dt", axis=1, inplace=True)

bitcoin.dropna(inplace=True)
bitcoin.head()

y = bitcoin.target
X = bitcoin.drop("target", axis=1)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import max_error, mean_absolute_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Max Error =", max_error(y_test, y_pred))
print("MAE =", mean_absolute_error(y_test, y_pred))
print("R2 =", r2_score(y_test, y_pred))

model = LinearRegression(positive=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Max Error =", max_error(y_test, y_pred))
print("MAE =", mean_absolute_error(y_test, y_pred))
print("R2 =", r2_score(y_test, y_pred))


model = RandomForestRegressor(max_depth=10,random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Max Error =", max_error(y_test, y_pred))
print("MAE =", mean_absolute_error(y_test, y_pred))
print("R2 =", r2_score(y_test, y_pred))


from sklearn.model_selection import GridSearchCV

model = RandomForestRegressor(random_state=42)

params = {
    "n_estimators": [50, 100, 500],
    "max_depth": [3, 10],
    "min_samples_split": [2, 4]
}
gs = GridSearchCV(model, params, scoring='neg_mean_squared_error')
gs.fit(X_train, y_train)

gs.best_params_

gs.best_score_

best_model = gs.best_estimator_

import pickle 

f = open("rfr.model", "wb")
pickle.dump(best_model, f)

my_model_file = open("rfr.model", "rb")
my_model = pickle.load(my_model_file)
my_model


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

print(train_data.head())
print(train_data.shape)
print(train_data.dtypes)

# checking nulls early so i know what im dealing with
null_counts = train_data.isnull().sum()
print(null_counts[null_counts > 0])

# first tried with more columns but keeping 3 for now
# features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt']
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']

X = train_data[features]
y = train_data['SalePrice']

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

# 80-20 split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_tr.shape[0])
print("Validation samples:", X_val.shape[0])

lr = LinearRegression()
lr.fit(X_tr, y_tr)

# model coefficients just to see what it learned
print("Coefficients:", lr.coef_)
print("Intercept:", lr.intercept_)

y_pred = lr.predict(X_val)

mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("MAE :", mae)
print("R2 Score :", r2)

# r2 is around 0.5, not bad for just 3 features
result = pd.DataFrame({
    "Actual": y_val.values,
    "Predicted": y_pred,
    "Difference": y_val.values - y_pred
})

print(result.head(10))

plt.scatter(y_val, y_pred, alpha=0.5, color='steelblue')

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted House Prices")

plt.tight_layout()
plt.show()

# manually testing with a sample house
sample_house = [[2000, 3, 2]]

estimated = lr.predict(
    pd.DataFrame(sample_house, columns=features)
)

print(
    "Estimated price for 2000sqft, 3bed, 2bath:",
    estimated[0]
)

# correlation, GrLivArea was around 0.7 with SalePrice
corr_matrix = train_data[
    ['SalePrice', 'GrLivArea', 'BedroomAbvGr', 'FullBath']
].corr()

print(corr_matrix)

avg = train_data['SalePrice'].mean()
median = train_data['SalePrice'].median()

print("avg price:", avg)
print("median price:", median)

# distribution is right skewed, most houses are in lower range
plt.hist(
    train_data['SalePrice'],
    bins=20,
    color='steelblue',
    edgecolor='black'
)

plt.xlabel("Sale Price")
plt.ylabel("Count")

plt.title("Sale Price Distribution")

plt.tight_layout()
plt.show()

# predicting on actual test set
test_features = test_data[features]

test_pred = lr.predict(test_features)

final_output = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice": test_pred
})

final_output.to_csv("submission.csv", index=False)

print("saved submission.csv")
print("done")
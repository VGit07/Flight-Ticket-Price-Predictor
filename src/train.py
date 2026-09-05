import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor as RF
import pickle as pkl
from prettytable import PrettyTable as PT

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("../data/data.csv")


# -----------------------------
# Select required columns
# -----------------------------

df = df[
    [
        "source_city",
        "destination_city",
        "airline",
        "class",
        "duration",
        "price"
    ]
]


# -----------------------------
# Rename columns
# -----------------------------

df.rename(
    columns={
        "source_city": "from",
        "destination_city": "to"
    },
    inplace=True
)


# -----------------------------
# Filter flights <= 5 hours
# removing vistara
# -----------------------------
df = df[df["duration"] <= 5]
df = df[df["airline"] != "Vistara"]
df = df[df["airline"] != "GO_FIRST"]


# -----------------------------
# Separate X and y
# -----------------------------
X = df.drop("price", axis=1)
y = df["price"]


# -----------------------------
# One-Hot Encoding
# -----------------------------
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

encoded = encoder.fit_transform(
    X[["from", "to", "airline", "class"]]
)


# Give encoded columns meaningful names
encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(
        ["from", "to", "airline", "class"]
    ),
    index=X.index
)


# Add duration
X_encoded = pd.concat(
    [
        encoded_df,
        X[["duration"]]
    ],
    axis=1
)


# Make sure all column names are strings
X_encoded.columns = X_encoded.columns.astype(str)


# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42
)



# -----------------------------
# Random Forest Model
# -----------------------------
model = RF(n_estimators=50,
           max_depth=20,
           min_samples_split=2)


# For showing Accuracy
table = PT()
table.field_names = ["Dataset","R² Score","MAE","RSME"]

# -----------------------------
# Test Data Accuracy
# -----------------------------
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

r2 = round(r2_score(y_test, y_pred),4)
mae = round(mean_absolute_error(y_test, y_pred),4)
rmse = round(mean_squared_error(y_test, y_pred) ** 0.5,4)
table.add_row(["Testing",r2,mae,rmse])

# -----------------------------
# Train Set Evaluation
# -----------------------------

model.fit(X_train,y_train)
y_pred = model.predict(X_train)

train_r2 = round(r2_score(y_train, y_pred),4)
train_mae = round(mean_absolute_error(y_train, y_pred),4)
train_rmse = round(mean_squared_error(y_train, y_pred) ** 0.5,4)
table.add_row(["Training",train_r2,train_mae,train_rmse])

print("-"*50)
print("Accuracy Score Table (Random Forest Model)")
print("-"*50)
print(table)
print("-"*50)

with open("../model/model.pkl","wb") as file:
    pkl.dump(model,file=file)

with open("../model/encoder.pkl","wb") as file:
    pkl.dump(encoder,file=file)
    
print("The Model & Encoder is save !!!!!")
print("-"*50)

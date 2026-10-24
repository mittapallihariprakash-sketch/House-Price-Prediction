from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load dataset
data_path = Path(__file__).parent / "data set" / "house_prices.csv"

if not data_path.exists():
    raise FileNotFoundError(
        f"CSV file not found: {data_path}\n"
        "Check the data set folder and CSV filename."
    )

df = pd.read_csv(data_path)
df.columns = df.columns.str.strip().str.lower()

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Clean dataset
df = df.dropna().drop_duplicates()

if "price" not in df.columns:
    raise ValueError("Dataset must contain a 'price' column.")

# 3. Separate features and target
X = df.drop(columns=["price"])
y = pd.to_numeric(df["price"], errors="coerce")

# Convert categorical columns into numeric columns
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Remove rows with invalid target values
valid_rows = y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows]

if len(df) < 5 or len(X) < 5:
    raise ValueError("Dataset has too few valid rows for training.")

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Evaluate model
y_pred = model.predict(X_test)

print("\n--- Model Performance ---")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):,.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")

# 7. Predict price for a sample from the dataset
sample_house = X_test.iloc[[0]]
predicted_price = model.predict(sample_house)[0]

print(f"\nActual House Price: {y_test.iloc[0]:,.2f}")
print(f"Predicted House Price: {predicted_price:,.2f}")
import joblib

joblib.dump(model, "model.joblib")
joblib.dump(X.columns.tolist(), "features_name.joblib")

print("Model and features saved successfully!")
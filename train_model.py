# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load Data
print("Loading data...")
df = pd.read_csv("customer_booking.csv", encoding="latin1")

# 2. Cleaning & Preprocessing (Sesuai alaika_datnal.ipynb)
# Mapping Hari
day_mapping = {
    "Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, 
    "Fri": 5, "Sat": 6, "Sun": 7
}
df['flight_day'] = df['flight_day'].map(day_mapping)

# Encoding Categorical Columns
cat_cols = ['sales_channel', 'trip_type', 'route', 'booking_origin']
le_dict = {}

print("Encoding categorical data...")
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le  # Simpan encoder untuk dipakai di app nanti

# Scaling Numerical Columns
num_cols = ['purchase_lead', 'length_of_stay', 'flight_duration']
scaler = StandardScaler()
print("Scaling numerical data...")
df[num_cols] = scaler.fit_transform(df[num_cols].astype(float))

# 3. Define X and y
X = df.drop('booking_complete', axis=1)
y = df['booking_complete']

# 4. Train Model
print("Training Random Forest Model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluasi Singkat
y_pred = rf_model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

# 5. Save Artifacts
print("Saving model and tools...")
joblib.dump(rf_model, 'airline_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(le_dict, 'encoders.joblib')

print("Selesai! File .joblib telah dibuat.")

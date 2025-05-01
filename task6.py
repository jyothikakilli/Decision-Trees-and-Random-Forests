import pandas as pd

# Load dataset
df = pd.read_csv("task6.csv")

# Print column names to verify
print("Existing columns:", df.columns)

# Rename columns only if the length matches
expected_columns = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", 
                    "FastingBS", "RestingECG", "MaxHeartRate", "ExerciseAngina",
                    "STDepression"]

if "HeartDisease" in df.columns:
    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]
else:
    print("Column 'HeartDisease' not found! Verify dataset structure.")

# Display updated DataFrame structure
print(df.head())

# Display dataset info
print(df.head())

# Split the data into features (X) and target (y)
X = df.drop(columns=["HeartDisease"])  # Dropping target column
y = df["HeartDisease"]  # Target variable

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Predictions
dt_pred = dt_model.predict(X_test)

# Evaluate the model
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("Decision Tree Classification Report:\n", classification_report(y_test, dt_pred))

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
rf_pred = rf_model.predict(X_test)

# Evaluate the model
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset from a CSV file
df = pd.read_csv("C:/Users/pargat/Desktop/B Data Science Internship/Data Glacier/Week 4/Week4_Deployment_on_Flask/iris.data.csv")

# Display the first few rows of the dataset to understand its structure
print(df.head())

# Select independent variables (features) and the dependent variable (target)
X = df[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]  # Features
y = df["Species"]  # Target variable (species)

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Feature scaling: Standardizing the features (mean=0, std=1) for better model performance
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Fit on training data and transform it
X_test = sc.transform(X_test)  # Transform the test data based on the training data scaling

# Instantiate the model
model = RandomForestClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

# Save the trained model to a file using joblib for later use
joblib.dump(model, 'iris_model.pkl')

# Print confirmation message
print("Model saved as iris_model.pkl")

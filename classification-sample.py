import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create a DataFrame from the dataset
data = {
    'Transaction Amount ($)': [1000, 500, 200, 3000, 150, 700, 1800],
    'Merchant Category': ['Retail', 'Restaurant', 'Online Shopping', 'Retail', 'Grocery', 'Gas Station', 'Retail'],
    'Time of Transaction (Hour)': [10, 18, 15, 12, 9, 16, 20],
    'Day of Week': ['Monday', 'Wednesday', 'Friday', 'Tuesday', 'Saturday', 'Thursday', 'Monday'],
    'Is Fraudulent': ['No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)


# Encoding categorical variables
df['Merchant Category'] = df['Merchant Category'].astype('category').cat.codes
df['Day of Week'] = df['Day of Week'].astype('category').cat.codes
df['Is Fraudulent'] = df['Is Fraudulent'].map({'Yes': 1, 'No': 0})

# Split the data into features (X) and target variable (y)
X = df.drop('Is Fraudulent', axis=1)
y = df['Is Fraudulent']

# Create and train the classification model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Generate predictions for the test set
new_transaction = pd.DataFrame({
    'Transaction Amount ($)': [1200],
    'Merchant Category': ['Online Shopping'],
    'Time of Transaction (Hour)': [14],
    'Day of Week': ['Monday']
})

# Encoding categorical variables
new_transaction['Merchant Category'] = new_transaction['Merchant Category'].astype('category').cat.codes
new_transaction['Day of Week'] = new_transaction['Day of Week'].astype('category').cat.codes

# Make predictions
prediction = model.predict(new_transaction)

# Display the prediction
print("Is the transaction fraudulent? Predicted:", "Yes" if prediction[0] == 1 else "No")

import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the accident dataset
data = pd.read_csv('accident_data.csv')

# Separate independent and dependent variables
independent_variables = data[['Weather', 'Road_Type', 'Time_of_Day', 'Driver_Age', 'Driver_Experience', 'Vehicle_Type']]
dependent_variable = data['Accident_Severity']

# Create and fit the linear regression model
model = LinearRegression()
model.fit(independent_variables, dependent_variable)

# Save the model for future use
import pickle
pickle.dump(model, open('accident_severity_model.pkl', 'wb'))

# Predict accident severity for a hypothetical set of independent variables
new_data = {
    'Weather': 'Rainy',
    'Road_Type': 'Urban',
    'Time_of_Day': 'Nighttime',
    'Driver_Age': 25,
    'Driver_Experience': 5,
    'Vehicle_Type': 'Car'
}

new_data_formatted = pd.DataFrame([new_data])
predicted_severity = model.predict(new_data_formatted)[0]
print("Predicted accident severity:", predicted_severity)

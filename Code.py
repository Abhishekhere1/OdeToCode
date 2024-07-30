### Code Snippets


# Load dataset
data = pd.read_csv('appointment_data.csv')

# Preprocess data
data['no_show'] = data['status'].apply(lambda x: 1 if x == 'No-Show' else 0)
features = data[['appointment_time', 'doctor_id', 'patient_history']]
labels = data['no_show']

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


#### AI Model Development

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')


#### Real-Time Schedule Adjustment

def adjust_schedule(appointments, model):
    current_time = datetime.datetime.now()
    for appointment in appointments:
        if model.predict([appointment['features']])[0] == 1:
            appointment['status'] = 'Cancelled'
        elif appointment['time'] < current_time:
            appointment['status'] = 'Missed'
    return appointments


# Sample appointments data
appointments = [
    {'features': [10, 1, 0], 'time': datetime.datetime(2024, 7, 28, 10, 0), 'status': 'Scheduled'},
    {'features': [11, 2, 1], 'time': datetime.datetime(2024, 7, 28, 11, 0), 'status': 'Scheduled'}
]

adjusted_appointments = adjust_schedule(appointments, model)
print(adjusted_appointments)


#### Predictive Analytics for Demand Forecasting

# Load and preprocess data
demand_data = pd.read_csv('demand_data.csv')
demand_series = demand_data['demand']

# Train model
model = ExponentialSmoothing(demand_series, seasonal='add', seasonal_periods=12).fit()

# Forecast demand
forecast = model.forecast(steps=12)
print(forecast)



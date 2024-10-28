from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from preprocessing import load_data, clean_data, feature_engineering, split_data

def train_random_forest(X_train, y_train):
    # Train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    df = load_data('data/sales_data.csv')
    df = clean_data(df)
    df = feature_engineering(df)
    
    X_train, X_test, y_train, y_test = split_data(df, target_column='Total_Sales')
    
    model = train_random_forest(X_train, y_train)
    evaluate_model(model, X_test, y_test)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    df.dropna(inplace=True)  # Dropping missing values
    return df

def feature_engineering(df):
    df['Total_Sales'] = df['Quantity'] * df['Price']  # Assuming Quantity and Price columns exist
    return df

def split_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardizing data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Example usage:
if __name__ == "__main__":
    df = load_data('data/sales_data.csv')
    df = clean_data(df)
    df = feature_engineering(df)

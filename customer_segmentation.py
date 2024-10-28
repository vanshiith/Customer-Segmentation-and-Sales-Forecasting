
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import load_data, clean_data, feature_engineering

def customer_segmentation(df, n_clusters=5):
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['Total_Sales', 'Quantity']])  # Example features
    return df

def visualize_clusters(df):
    # Visualizing clusters
    plt.scatter(df['Total_Sales'], df['Quantity'], c=df['Cluster'], cmap='viridis')
    plt.xlabel('Total Sales')
    plt.ylabel('Quantity')
    plt.title('Customer Segmentation')
    plt.show()

if __name__ == "__main__":
    df = load_data('data/sales_data.csv')
    df = clean_data(df)
    df = feature_engineering(df)
    
    df = customer_segmentation(df)
    visualize_clusters(df)

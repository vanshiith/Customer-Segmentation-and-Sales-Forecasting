
import pandas as pd
import plotly.express as px
from preprocessing import load_data, clean_data, feature_engineering

def generate_basic_visualization(df):
    fig = px.scatter(df, x='Total_Sales', y='Quantity', color='Cluster',
                     title='Customer Segmentation based on Sales and Quantity')
    fig.show()

def export_to_tableau(df):
    df.to_csv('data/segmentation_output.csv', index=False)

if __name__ == "__main__":
    df = load_data('data/sales_data.csv')
    df = clean_data(df)
    df = feature_engineering(df)
  
    generate_basic_visualization(df)
    export_to_tableau(df)

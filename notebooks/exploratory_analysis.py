#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exploratory Data Analysis - Favorita Store Sales

This script contains the exploratory analysis of data from the 
Store Sales Forecasting competition for Favorita stores in Ecuador.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette('viridis')

# Path definitions
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_DIR / 'data' / 'raw'
REPORTS_DIR = PROJECT_DIR / 'reports' / 'figures'

# Create directory for figures if it doesn't exist
os.makedirs(REPORTS_DIR, exist_ok=True)

print(f"Project directory: {PROJECT_DIR}")
print(f"Raw data directory: {RAW_DATA_DIR}")
print(f"Figures directory: {REPORTS_DIR}")

# Function to load data
def load_data():
    datasets = {}
    files = [
        "train.csv", 
        "test.csv", 
        "holidays_events.csv", 
        "oil.csv", 
        "stores.csv", 
        "transactions.csv"
    ]
    
    for file in files:
        file_path = RAW_DATA_DIR / file
        if file_path.exists():
            print(f"Loading {file}...")
            datasets[file.split('.')[0]] = pd.read_csv(file_path)
        else:
            print(f"File not found: {file_path}")
    
    return datasets

# Analysis of training dataset
def analyze_train_data(train_df):
    # Convert date column to datetime format
    train_df['date'] = pd.to_datetime(train_df['date'])
    
    # Descriptive statistics of sales
    print("\nDescriptive statistics of sales:")
    print(train_df['sales'].describe())
    
    # Display first records
    print("\nFirst records:")
    print(train_df.head())
    
    # Distribution of sales
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(train_df['sales'], kde=True)
    plt.title('Sales Distribution')
    plt.xlabel('Sales')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    sns.histplot(np.log1p(train_df['sales']), kde=True)
    plt.title('Log Sales Distribution')
    plt.xlabel('Log(Sales+1)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'sales_distribution.png')
    plt.close()
    
    # Aggregation of sales by date
    daily_sales = train_df.groupby('date')['sales'].sum().reset_index()
    
    # Time series plot
    plt.figure(figsize=(16, 6))
    plt.plot(daily_sales['date'], daily_sales['sales'])
    plt.title('Total Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'sales_time_series.png')
    plt.close()
    
    # Analysis by day of week
    train_df['day_of_week'] = train_df['date'].dt.dayofweek
    train_df['day_name'] = train_df['date'].dt.day_name()
    
    weekday_sales = train_df.groupby('day_name')['sales'].mean().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=weekday_sales.index, y=weekday_sales.values)
    plt.title('Average Sales by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Sales')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'sales_by_weekday.png')
    plt.close()
    
    return daily_sales

# Promotion analysis
def analyze_promotions(train_df):
    # Comparison of sales with and without promotion
    train_df['onpromotion'] = train_df['onpromotion'].astype(bool)
    promo_sales = train_df.groupby('onpromotion')['sales'].mean()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=['No Promotion', 'With Promotion'], y=promo_sales.values)
    plt.title('Average Sales by Promotion Status')
    plt.xlabel('Promotion Status')
    plt.ylabel('Average Sales')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'sales_by_promotion.png')
    plt.close()

# Analysis by product family
def analyze_product_families(train_df):
    # Average sales by product family
    family_sales = train_df.groupby('family')['sales'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x=family_sales.index[:15], y=family_sales.values[:15])
    plt.title('Top 15 Product Families by Average Sales')
    plt.xlabel('Product Family')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'sales_by_product_family.png')
    plt.close()

# Analysis by store
def analyze_stores(train_df, stores_df):
    # Aggregation of sales by store
    store_sales = train_df.groupby('store_nbr')['sales'].mean().reset_index()
    store_info = pd.merge(store_sales, stores_df, on='store_nbr')
    
    # Visualize average sales by store type
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='type', y='sales', data=store_info)
    plt.title('Sales Distribution by Store Type')
    plt.xlabel('Store Type')
    plt.ylabel('Average Sales')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'sales_by_store_type.png')
    plt.close()
    
    # Average sales by city
    city_sales = store_info.groupby('city')['sales'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(14, 6))
    sns.barplot(x=city_sales.index[:10], y=city_sales.values[:10])
    plt.title('Top 10 Cities by Average Sales')
    plt.xlabel('City')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'sales_by_city.png')
    plt.close()

# Holiday analysis
def analyze_holidays(train_df, holidays_df, daily_sales):
    # Convert date column to datetime format
    holidays_df['date'] = pd.to_datetime(holidays_df['date'])
    
    # Filter only national holidays and not transferred
    national_holidays = holidays_df[
        (holidays_df['locale'] == 'National') & (holidays_df['transferred'] == False)
    ]
    
    # Create a DataFrame with dates and holiday indicator
    all_dates = pd.DataFrame({'date': pd.date_range(
        start=train_df['date'].min(), 
        end=train_df['date'].max()
    )})
    all_dates['is_holiday'] = all_dates['date'].isin(national_holidays['date']).astype(int)
    
    # Merge with daily sales
    daily_sales_with_holidays = pd.merge(daily_sales, all_dates, on='date', how='left')
    
    # Compare sales on holidays vs. normal days
    holiday_impact = daily_sales_with_holidays.groupby('is_holiday')['sales'].mean()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=['Normal Day', 'National Holiday'], y=holiday_impact.values)
    plt.title('Impact of National Holidays on Sales')
    plt.xlabel('Day Type')
    plt.ylabel('Average Sales')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'sales_by_holidays.png')
    plt.close()

# Oil price analysis
def analyze_oil_prices(oil_df, daily_sales):
    # Convert date column to datetime format
    oil_df['date'] = pd.to_datetime(oil_df['date'])
    
    # Fill missing values
    oil_df = oil_df.sort_values('date')
    oil_df['dcoilwtico'] = oil_df['dcoilwtico'].fillna(method='ffill')
    
    # Merge with daily sales
    sales_with_oil = pd.merge(daily_sales, oil_df, on='date', how='left')
    sales_with_oil['dcoilwtico'] = sales_with_oil['dcoilwtico'].fillna(method='ffill')
    
    # Visualization of the relationship between oil price and sales
    plt.figure(figsize=(16, 8))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(sales_with_oil['dcoilwtico'], sales_with_oil['sales'], alpha=0.5)
    plt.title('Relationship Between Oil Price and Sales')
    plt.xlabel('Oil Price (WTI)')
    plt.ylabel('Total Sales')
    plt.grid(True)
    
    # Time trend
    plt.subplot(1, 2, 2)
    plt.plot(sales_with_oil['date'], sales_with_oil['sales'], label='Sales')
    plt.plot(sales_with_oil['date'], sales_with_oil['dcoilwtico'] * 1000, 
             label='Oil Price (scaled)', alpha=0.7)
    plt.title('Time Trend: Sales vs. Oil Price')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'sales_oil_price.png')
    plt.close()

# Main function
def main():
    print("Starting exploratory data analysis...\n")
    
    # Load data
    data = load_data()
    
    if 'train' not in data:
        print("Error: Training data not found!")
        return
    
    # Run analyses
    daily_sales = analyze_train_data(data['train'])
    
    analyze_promotions(data['train'])
    
    analyze_product_families(data['train'])
    
    if 'stores' in data:
        analyze_stores(data['train'], data['stores'])
    
    if 'holidays_events' in data:
        analyze_holidays(data['train'], data['holidays_events'], daily_sales)
    
    if 'oil' in data:
        analyze_oil_prices(data['oil'], daily_sales)
    
    print("\nExploratory analysis completed! Visualizations saved at:", REPORTS_DIR)

if __name__ == "__main__":
    main() 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os
import warnings

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('Train.csv')

# Step 1: Assign priority to categories
# Assuming priorities based on some business logic (example)
category_priority = {1: 'High', 2: 'Medium', 3: 'Low'}
df['CATEGORY_PRIORITY'] = df['CATEGORY_ID'].map(category_priority)

# Reorder based on CATEGORY_ID priority (ascending order)
df = df.sort_values(by='CATEGORY_ID')

# Step 2: Scaling and Standardizing Numerical Features (e.g., ENTITY_LENGTH)
scaler = StandardScaler()
df[['ENTITY_LENGTH']] = scaler.fit_transform(df[['ENTITY_LENGTH']])

# Ensure the directory exists to save the plots
output_dir = "EDA_Plots"
os.makedirs(output_dir, exist_ok=True)

# Step 3: Separate and Calculate EDA Metrics for Each Category
for category in df['CATEGORY_ID'].unique():
    print(f"\nCategory: {category}")
    df_category = df[df['CATEGORY_ID'] == category]

    # 1. Summary Statistics for the Category
    print(f"\nSummary Statistics for Category {category}:")
    print(df_category.describe())

    # 2. Check Data Types & Missing Values
    print(f"\nData Types for Category {category}:")
    print(df_category.dtypes)

    print(f"\nMissing Values in Category {category}:")
    print(df_category.isnull().sum())

    # 3. Distribution of Numerical Features for the Category
    plt.figure(figsize=(10, 6))
    df_category.hist(column=['ENTITY_LENGTH'], bins=30, edgecolor='black')
    plt.suptitle(f'Histograms for Category {category} (Numerical Columns)', fontsize=14)
    plt.savefig(f'{output_dir}/histogram_category_{category}.png')  # Save histogram
    plt.close()

    # 4. Boxplot for Outliers in Numerical Features
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_category[['ENTITY_LENGTH']])
    plt.title(f'Boxplot for ENTITY_LENGTH (Category {category})')
    plt.savefig(f'{output_dir}/boxplot_entity_length_category_{category}.png')  # Save boxplot
    plt.close()

    # 5. Correlation Matrix for Numerical Features
    correlation_matrix = df_category.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', linewidths=0.5)
    plt.title(f'Correlation Matrix for Category {category}')
    plt.savefig(f'{output_dir}/correlation_matrix_category_{category}.png')  # Save correlation heatmap
    plt.close()

    # 6. Outlier Detection using Z-score for the Category
    z_scores = np.abs(stats.zscore(df_category[['ENTITY_LENGTH']]))
    outliers = np.where(z_scores > 3)
    print(f"\nOutliers Detected in Category {category}:")
    print(df_category.iloc[outliers[0]])

    # 7. Feature Relationships with Target Variable (Boxplot/Violin Plot)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='CATEGORY_ID', y='ENTITY_LENGTH', data=df_category)
    plt.title(f'Boxplot: ENTITY_LENGTH vs CATEGORY_ID (Category {category})')
    plt.savefig(f'{output_dir}/boxplot_relationship_category_{category}.png')  # Save boxplot for relationship
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.violinplot(x='CATEGORY_ID', y='ENTITY_LENGTH', data=df_category)
    plt.title(f'Violin Plot: ENTITY_LENGTH vs CATEGORY_ID (Category {category})')
    plt.savefig(f'{output_dir}/violinplot_relationship_category_{category}.png')  # Save violin plot
    plt.close()

print("\nAnalysis complete for all categories! Plots saved in the directory.")
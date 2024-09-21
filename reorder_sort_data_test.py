import pandas as pd

# Load the dataset
df = pd.read_csv('Test.csv')

# Define priority for CATEGORY_ID (lower number = higher priority)
priority = {
    30: 1,    # High priority categories
    112: 2,
    2201: 3,
    6104: 4,
    8360: 5,  # Lowest priority
}

# Add a priority column based on CATEGORY_ID
df['PRIORITY'] = df['CATEGORY_ID'].map(priority)

# Replace any missing priorities with a default low priority value
df['PRIORITY'] = df['PRIORITY'].fillna(len(priority) + 1)

# Sort by priority and then by ENTITY_ID for further ordering
df_sorted = df.sort_values(by=['PRIORITY', 'ENTITY_ID'])

# Remove the priority column (if not needed for final dataset)
df_sorted = df_sorted.drop(columns=['PRIORITY'])

# Group by CATEGORY_ID for clear category separation (optional)
grouped_df = df_sorted.groupby('CATEGORY_ID')

# Save or display the sorted dataset
df_sorted.to_csv('Sorted_Train.csv', index=False)
df_sorted.head()  # Display the first few rows of the sorted data
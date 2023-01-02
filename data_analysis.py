import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Generation (for demonstration purposes)
def generate_sample_data(num_samples=100):
    np.random.seed(42)
    data = {
        'Feature1': np.random.rand(num_samples) * 100,
        'Feature2': np.random.rand(num_samples) * 50,
        'Category': np.random.choice(['A', 'B', 'C'], num_samples),
        'Target': np.random.rand(num_samples) * 200 + 50
    }
    df = pd.DataFrame(data)
    df['Target'] = df['Target'] + df['Feature1'] * 0.5 - df['Feature2'] * 0.2
    df['Target'] = df['Target'].astype(int)
    return df

# 2. Exploratory Data Analysis (EDA)
def perform_eda(df):
    print("\n--- Exploratory Data Analysis ---")
    print("Shape:", df.shape)
    print("\nInfo:")
    df.info()
    print("\nDescription:")
    print(df.describe())
    print("\nValue Counts for Category:")
    print(df['Category'].value_counts())

# 3. Data Visualization
def visualize_data(df):
    print("\n--- Data Visualization ---")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='Feature1', y='Target', hue='Category', data=df)
    plt.title('Feature1 vs Target by Category')
    
    plt.subplot(1, 2, 2)
    sns.histplot(df['Target'], kde=True)
    plt.title('Distribution of Target Variable')
    
    plt.tight_layout()
    plt.savefig('target_distribution.png')
    plt.close()
    print("Saved target_distribution.png")

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Category', y='Target', data=df)
    plt.title('Target Distribution by Category')
    plt.savefig('target_by_category.png')
    plt.close()
    print("Saved target_by_category.png")

# 4. Main Execution
if __name__ == '__main__':
    # Generate sample data
    data_frame = generate_sample_data(num_samples=200)
    
    # Perform EDA
    perform_eda(data_frame)
    
    # Visualize data
    visualize_data(data_frame)
    
    print("\nData analysis and visualization complete.")

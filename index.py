# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load and Explore the Dataset
def load_and_explore():
    try:
        # Load the Iris dataset from sklearn
        iris_data = load_iris()
        df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
        df['species'] = iris_data.target
        
        # Map target values to actual species names
        df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        # Display the first few rows of the DataFrame
        print(df.head())
        
        # Check the data types and missing values
        print(df.info())
        print(df.isnull().sum())
        
        # Clean the dataset (no missing values in this dataset, but here's how you could do it)
        df.fillna(df.mean(), inplace=True)  # Example of filling missing values
        df.dropna(inplace=True)  # Example of dropping missing values

        return df

    except Exception as e:
        print(f"An error occurred: {e}")

# Basic Data Analysis
def basic_data_analysis(df):
    # Compute basic statistics
    print(df.describe())
    
    # Compute the mean of numerical columns grouped by species
    grouped_means = df.groupby('species').mean()
    print(grouped_means)
    
    return grouped_means

# Data Visualization
def data_visualization(df):
    # Line chart (for illustrative purposes, creating a cumulative sum)
    df['cumulative_sum'] = df['sepal length (cm)'].cumsum()
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['cumulative_sum'])
    plt.title('Cumulative Sum of Sepal Length Over Index')
    plt.xlabel('Index')
    plt.ylabel('Cumulative Sum of Sepal Length (cm)')
    plt.show()
    
    # Bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x='species', y='sepal length (cm)', data=df)
    plt.title('Average Sepal Length per Species')
    plt.xlabel('Species')
    plt.ylabel('Average Sepal Length (cm)')
    plt.show()
    
    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['petal length (cm)'], bins=20, edgecolor='black')
    plt.title('Distribution of Petal Length')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Frequency')
    plt.show()
    
    # Scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
    plt.title('Sepal Length vs. Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend(title='Species')
    plt.show()

# Main execution
if __name__ == "__main__":
    df = load_and_explore()
    if df is not None:
        basic_data_analysis(df)
        data_visualization(df)

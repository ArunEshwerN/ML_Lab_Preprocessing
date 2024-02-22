import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Function to plot and save histograms
def plot_and_save_histograms(dataframe, features, phase):
    for feature in features:
        plt.figure(figsize=(8, 4))
        sns.histplot(dataframe[feature], kde=False, stat="count", bins=30)
        plt.title(f'{phase} Histogram of {feature}')
        plt_path = f'histogram_{feature}_{phase}.png'
        plt.savefig(plt_path)
        plt.close()

# Function to plot and save distribution plots with density estimation
def plot_and_save_distribution_plots(dataframe, features, phase):
    for feature in features:
        plt.figure(figsize=(8, 4))
        sns.kdeplot(dataframe[feature], fill=True)
        plt.title(f'{phase} Distribution Plot of {feature}')
        plt_path = f'distribution_{feature}_{phase}.png'
        plt.savefig(plt_path)
        plt.close()

# Function to plot and save missing values heatmap
def plot_and_save_missing_values(dataframe):
    plt.figure(figsize=(10, 6))
    sns.heatmap(dataframe.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt_path = 'missing_values_heatmap.png'
    plt.savefig(plt_path)
    plt.close()

# Function to plot and save boxplots for outlier detection and post-handling
def plot_and_save_boxplots(dataframe, features, phase):
    for feature in features:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=dataframe[feature])
        plt.title(f'{phase} Boxplot of {feature}')
        plt_path = f'boxplot_{feature}_{phase}.png'
        plt.savefig(plt_path)
        plt.close()

# Load the dataset
df = pd.read_csv('avocado.csv')

# Initial visualizations before preprocessing
plot_and_save_histograms(df, df.select_dtypes(include=[np.number]).columns, 'Before')
plot_and_save_distribution_plots(df, df.select_dtypes(include=[np.number]).columns, 'Before')
plot_and_save_missing_values(df)
plot_and_save_boxplots(df, df.select_dtypes(include=[np.number]).columns, 'Before')

# Outlier removal
numeric_features = df.select_dtypes(include=[np.number]).columns
for feature in numeric_features:
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    df = df[(df[feature] >= q1 - 1.5 * iqr) & (df[feature] <= q3 + 1.5 * iqr)]

# Visualizations after outlier handling
plot_and_save_boxplots(df, numeric_features, 'After')

# Encoding categorical variables
categorical_features = df.select_dtypes(include=['object']).columns.drop('Date')
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Scaling numerical features
scaler = MinMaxScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Final visualizations after preprocessing
plot_and_save_histograms(df, numeric_features, 'After')
plot_and_save_distribution_plots(df, numeric_features, 'After')

# Splitting the dataset
X = df.drop('AveragePrice', axis=1)  # Assuming AveragePrice is the target
y = df['AveragePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Preprocessing complete. Visualizations saved.")

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path):
    df = None
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
        except Exception as e:
            print(f"Error loading data: {e}")
    return df


def handle_missing_values(df):
    for column in df.select_dtypes(include=[np.number]):
        df[column] = df[column].fillna(df[column].mean())
    for column in df.select_dtypes(include=[np.object]):
        df[column] = df[column].fillna(df[column].mode()[0])
    return df


def select_features(df, target_column, threshold=0.5):
    if target_column not in df.columns:
        raise KeyError(f"Target column {target_column} does not exist in DataFrame")
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].isnull().any():
                df[col] = df[col].astype(str).fillna('NaN')
            df[col] = le.fit_transform(df[col])
    correlated_features = df.corr()[target_column].sort_values(ascending=False)
    selected_columns = correlated_features.index[correlated_features.abs() > threshold]
    df = df[selected_columns]
    return df


def reduce_dimensions(df, n_dimensions=2):
    n_dimensions = min(n_dimensions, df.shape[1], df.shape[0])
    pca = PCA(n_components=n_dimensions)
    df = pd.DataFrame(pca.fit_transform(df), columns=[f"PC{i+1}" for i in range(n_dimensions)])
    return df


def preprocess_data(df):
    le = LabelEncoder()
    scaler = StandardScaler()
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        df[column] = df[column].astype(str)
        df[column] = le.fit_transform(df[column])
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for column in numerical_columns:
        df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
    return df


def visualize_data(df):
    if len(df.columns) > 1:
        df[df.columns[:2]].plot(kind='scatter', x=df.columns[0], y=df.columns[1])
        plt.show()
    df[df.columns[0]].hist()
    plt.show()

    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for column in numerical_columns:
        df.boxplot(column=[column])
        plt.show()
        df[column].plot.line()
        plt.show()
    if len(numerical_columns) > 1:
        corr = df[numerical_columns].corr()
        sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f")
        plt.show()


def main():
    file_path = input('Enter file path: ')
    df = load_data(file_path)
    if df is not None:
        print(df.head())
    df = handle_missing_values(df)
    print('Missing values handled')

    target_column = input('Enter target column: ')
    df = select_features(df, target_column)
    print('Feature selection done')

    df = reduce_dimensions(df)
    print('Dimensionality reduction done')

    df = preprocess_data(df)
    print('Data preprocessing done')

    visualize_data(df)


if __name__ == "__main__":
    main()


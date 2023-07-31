#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import plotly
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



st.sidebar.header('Help Menu')
with st.sidebar:
 
  button_clicked = st.button("**Help**")

  if button_clicked:
      st.write("Welcome to my Streamlit app!")
      st.write("This is a simple help menu.")
      st.write("Here are some instructions on how to use the app:")
      st.write("1. Upload your file.")
      st.write("2. Select a column .")
      st.write("3. Click on Visualize Data.")
      st.write("Enjoy using the app!")

image = Image.open('logoData.png')
st.image(image)

    #columns
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    # Add custom CSS style to position the button at the top-left corner
    st.markdown(
        """
        <style>
        .custom-button {
            position: absolute;s
            top: 10px;
            left: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    button_containers = st.empty()

    # Button to be placed at the top-left and in the middle
    if button_containers.button("**Contact us**", key="myycustom"):
        #st.write("Automated EDA App")
        st.write("khadidja_mek@hotmail.fr")

        
with col2:
    # Add custom CSS style to position the button at the top-left and in the middle
    st.markdown(
        """
        <style>
        .custom-button {
            position: absolute;
            top: 50%;
            left: 10px;
            transform: translateY(-50%);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Use st.empty() to create an empty container at the top-left and in the middle
    button_container = st.empty()
    button_container = st.empty()
    if button_container.button("**Dataset**", key="Data"):
        st.write("https://www.kaggle.com/datasets/parulpandey/us-international-air-traffic-data")
        
    
with col3:
    # Add custom CSS style to position the button at the top-left corner
    st.markdown(
        """
        <style>
        .custom-button {
            position: absolute;
            top: 10px;
            left: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

        
with col5:
    # Add custom CSS style to position the button at the top-left corner
    st.markdown(
        """
        <style>
        .custom-button {
            position: absolute;
            top: 10px;
            left: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def load_data(uploaded_file):
    df = None
    try:
        # Try reading as CSV
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        try:
            # Try reading as Excel
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error loading data: {e}")

    return df


def handle_missing_values(df):
    # Numerical features: fill with mean
    for column in df.select_dtypes(include=[np.number]):
        df[column] = df[column].fillna(df[column].mean())

    # Categorical features: fill with mode
    for column in df.select_dtypes(include=[np.object]):
        df[column] = df[column].fillna(df[column].mode()[0])
    
    return df


from sklearn.preprocessing import LabelEncoder

def select_features(df, target_column, threshold=0.5):
    if target_column not in df.columns:
        raise KeyError(f"Target column {target_column} does not exist in DataFrame")
    
    # Create a label encoder object
    le = LabelEncoder()

    # Apply label encoding for each categorical column
    for col in df.columns:
        if df[col].dtype == 'object':
            # Handle case where the column has null values
            if df[col].isnull().any():
                df[col] = df[col].astype(str).fillna('NaN')
            df[col] = le.fit_transform(df[col])

    # Calculate correlation with target column
    correlated_features = df.corr()[target_column].sort_values(ascending=False)

    # Keep only highly correlated features
    selected_columns = correlated_features.index[correlated_features.abs() > threshold]
    df = df[selected_columns]
    
    return df


def reduce_dimensions(df, n_dimensions=2):
    # Assume you are reducing dimensions using PCA
    n_dimensions = min(n_dimensions, df.shape[1], df.shape[0])
    
    pca = PCA(n_components=n_dimensions)
    df = pd.DataFrame(pca.fit_transform(df), columns=[f"PC{i+1}" for i in range(n_dimensions)])
    
    return df


def preprocess_data(df):
    le = LabelEncoder()
    scaler = StandardScaler()

    # Selecting object (string) columns which are usually categorical
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        # Convert column to string type
        df[column] = df[column].astype(str)
        df[column] = le.fit_transform(df[column])

    # Selecting numeric columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for column in numerical_columns:
        df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))

    return df



def visualize_data(df):
    # If DataFrame has more than one column, create scatter plot
    if len(df.columns) > 1:
        fig = px.scatter(df, x=df.columns[0], y=df.columns[1])
        st.plotly_chart(fig)

    # Plot distribution for the first column
    st.write(f'Distribution of {df.columns[0]}')
    fig, ax = plt.subplots()
    df[df.columns[0]].hist(ax=ax)
    st.pyplot(fig)

    # Bar plot for categorical features
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        st.write(f'Bar plot of {column}')
        fig = px.bar(df, x=column, y=df.index, title=f'{column} distribution')
        st.plotly_chart(fig)

        # Pie chart for categorical features
        st.write(f'Pie chart of {column}')
        fig = px.pie(df, names=column, title=f'{column} distribution')
        st.plotly_chart(fig)

    # Box plots and line plots for numerical features
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for column in numerical_columns:
        st.write(f'Box plot of {column}')
        fig = px.box(df, y=column)
        st.plotly_chart(fig)

        st.write(f'Line plot of {column}')
        fig = px.line(df, y=column)
        st.plotly_chart(fig)

    # Correlation heatmap for numerical features
    if len(numerical_columns) > 1:
        st.write('Correlation Heatmap')
        fig, ax = plt.subplots(figsize=(10,8))
        corr = df[numerical_columns].corr()
        sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f")
        st.pyplot(fig)



def main():
    st.title('Automated EDA tool')

    # File uploader
    uploaded_file = st.file_uploader('Choose a CSV or Excel file', type=['csv', 'xlsx', 'xls'])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.dataframe(df)

        df = handle_missing_values(df)
        st.write('Missing values handled')
        
        target_column = st.selectbox('Select target column for feature selection', df.columns)
        df = select_features(df, target_column)
        st.write('Feature selection done')
        
        #df = reduce_dimensions(df)
        st.write('Dimensionality reduction done')
        
        df = preprocess_data(df)
        st.write('Data preprocessing done')

        if st.button('Visualize Data'):
            visualize_data(df)


if __name__ == "__main__":
    main()



st.subheader('Author👑')
st.write('**Khadidja Mekiri**' )

















    

    
        
        


















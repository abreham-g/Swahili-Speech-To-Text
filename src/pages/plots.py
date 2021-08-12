import streamlit as st
import awesome_streamlit as ast
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_dist(df: pd.DataFrame, column: str):
    plt.figure(figsize=(9, 7))
    sns.distplot(df).set_title(f'Distribution of {column}')
    plt.show()


def write():
    with st.spinner("Loading Plots ..."):
        st.title('Exploratory Data Analysis')

        # read the datasets
        na_value=['',' ','nan','Nan','NaN','na', '<Na>']
        df = pd.read_csv('src/pages/metadata.csv', na_values=na_value)

        #st.sidebar.title("Gallery")
        st.sidebar.subheader("Choose Feature to plot")
        plot = st.sidebar.selectbox("feature", ( "Durations",'Outlier','Translation Length','Character Per Seconds'))

        duration_df = df[['duration', 'filename']]
        if plot == 'Durations':
            print(df)
            plot_dist(duration_df, 'duration')
            st.pyplot()

            #plots.plot_hist(train, 'duration', 'orange')


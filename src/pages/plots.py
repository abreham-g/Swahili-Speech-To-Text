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
            st.header("Analysis of Audio Duration")
            st.subheader("Distribution of Duration")
            st.image('src/images/distribution_of _duration.png')
            st.write("""
            Most of the audio has a duration of between 2.25sec to 2.6sec.The max duration is 6.1s and the min duration is also 2.1s
            """)
            #print(df)
            #plot_dist(duration_df, 'duration')
            #st.pyplot()

            #plots.plot_hist(train, 'duration', 'orange')
        elif plot == 'Outlier':
            st.subheader("Outlier Detection for Duration")
            st.image('src/images/outliner_distribution.png')
            st.write("""
            #### The above plot show us we have no outlier in the audio data duration
            """)
        elif plot == 'Translation Length':
            st.header("Analysis of Audio Transcription")
            st.subheader("Distribution of Transcription Length")
            st.image('src/images/dist_of_trans_len.png')
            st.write("""
            Most of the audio has a transcription lenght of 40 characters. The max transcription length is 121characters  and the min transcription length is 5 characters
            """)
        elif plot == 'Character Per Seconds':
            st.subheader("Distribution of Character per sec")
            st.image('src/images\dist_char_per_sec.png')
        

# Libraries
import streamlit as st
import pandas as pd 
import awesome_streamlit as ast

def write():
    
    with st.spinner("Loading Data ..."):
        st.title('Data description  ')
        na_value=['',' ','nan','Nan','NaN','na', '<Na>']
        df = pd.read_csv('src/pages/metadata.csv', na_values=na_value)
        
       
        st.write(df.sample(20))
        print("succesfully loaded")
        st.title("the duration of audios")
        duration_df = df[['duration', 'filename']]
        st.write(duration_df.describe())
        st.write("the maximum duration of audios in our dataset is 6.1s")
        st.write(" the minimum duration of audios in our dataset is also 2.1s")
        st.title(" top 5 longest audios")
        long_audios = duration_df.sort_values(by="duration", ascending=False).head()
        st.write(long_audios.sample(5))
        st.title(" top 5 shortest audios")
        short_audios = duration_df.sort_values(by="duration", ascending=True).head()
        st.write(short_audios.sample(5))

        



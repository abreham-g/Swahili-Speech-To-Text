# libraries
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os


import warnings
warnings.filterwarnings(action="ignore")

        
        
        
def write():
    with st.spinner("Loading Data ..."):
        st.title('Speech To Text Predictions ')


        audio_file = st.file_uploader("upload Audio",type=['mp3','wav'])
        if st.button("Predict", key='predict'):
            with st.spinner("Loading Data ..."):
                st.title('Result ')
                st.write("""
                Predictions and the accuracy of the predictions.
                """)
                with open(audio_file.name,"wb") as f:
                    f.write(audio_file.getbuffer())
                    #saving 
                st.success("success")
                



  
    def load_preprocess_data():

        # load data
        na_value=['',' ','nan','Nan','NaN','na', '<Na>']
        store = pd.read_csv('src/pages/metadata.csv', na_values=na_value)
           
    # the models + predictions
    st.sidebar.title("Predictions")
    st.sidebar.subheader("Choose Model")
    regressor = st.sidebar.selectbox("Regressor", ("CNN and RNN", "RNN", "sequence to sequence  "))
    
  
    if regressor == 'CNN and RNN':

        metrics = st.sidebar.multiselect("What metrics to display?", (' loss', 'CC'))
 

 


import streamlit as st
import awesome_streamlit as ast



def write():
   
    with st.spinner("Loading Home ..."):
        st.title('Speech To Text')
        #st.image('../pic.png', use_column_width=True)
       
        st.write(
            """
           Speech recognition technology allows for hands-free control of smartphones, speakers,
and even vehicles in a wide variety of languages. Companies have moved towards the
goal of enabling machines to understand and respond to more and more of our
verbalized commands. There are many matured speech recognition systems available,
such as Google Assistant, Amazon Alexa, and Apple’s Siri. However, all of those voice
assistants work for limited languages only.
                """
        )

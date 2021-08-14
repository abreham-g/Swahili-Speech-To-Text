# import the child scripts
import streamlit as st
import awesome_streamlit as ast
import src.pages.home
import src.pages.data 
import src.pages.plots
import src.pages.transcribe



ast.core.services.other.set_logging_format()

# create the pages
PAGES = {
    "Home": src.pages.home,
    "Data":src.pages.data,
    "Data Exploration": src.pages.plots,
    "Predictions": src.pages.transcribe,

}


# render the pages
def main():
   
    st.sidebar.title("Speech To Text")
    selection = st.sidebar.radio("Select", list(PAGES.keys()))

    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        ast.shared.components.write_page(page)
    if selection =="Home":
        st.sidebar.title("INFORMATION")
        st.sidebar.info(
        """
The World Food Program wants to deploy an intelligent form that collects nutritional
information of food bought and sold at markets in two different countries in Africa -
Ethiopia and Kenya. The design of this intelligent form requires selected people to install
an app on their mobile phone, and whenever they buy food, they use their voice to
activate the app to register the list of items they just bought in their own language. The
intelligent systems in the app are expected to live to transcribe the speech-to-text and
organize the information in an easy-to-process way in a database.
        """
    )
    elif selection=="Predictions":
        st.sidebar.title("")


if __name__ == "__main__":
    main()


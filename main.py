"""This is the main module to run the app"""

import streamlit as st
from web_functions import load_data
# Configure the app
st.set_page_config(
    page_title = 'Prostate Cancer Detection',
    layout = 'wide',
    initial_sidebar_state = 'auto'
)


from Tabs import prostatecancer, lungcancer


Tabs = {
    "Lung Cancer": lungcancer,
    "Prostate Cancer": prostatecancer,
}


st.sidebar.title("Navigation")

page = st.sidebar.radio("Pages", list(Tabs.keys()))

df, X,y = load_data()

if page in ["Prostate Cancer"]:
    Tabs[page].app(df)
else:
    Tabs[page].app()
    

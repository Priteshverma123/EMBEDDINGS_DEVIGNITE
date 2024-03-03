import streamlit as st
from streamlit_lottie import st_lottie 
import requests
import json
from PIL import Image
from streamlit_option_menu import option_menu
import normal
import video
# Load image
im = Image.open("MEDI-ALERTAI_AT_WORK__1_-removebg-preview.png")

# Configure page settings
st.set_page_config(
    
    page_title="MEDI-ALERT",
    page_icon=im,
    layout="wide"
)

st.title("MEDI-ALERT: AI AT WORK!")

# Sidebar with option to select a page
st.sidebar.success("Select a page above")

# Additional sidebar for Normal Call and Video Call options
selected_page = st.sidebar.radio("Go to:", ["Home", "Normal Call", "Video Call"])



  

# Redirect based on selection
if selected_page == "Normal Call":
    st.write("Redirecting to Normal Call page...")
    normal.app()

elif selected_page == "Video Call":
    st.write("Redirecting to Video Call page...")
    video.app()

else:
    col1, col2 = st.columns([3, 2])

# Add text to the first column

    st.write("This web app functions as a Medical Emergency Response System (MERS).")
    st.write("Use the navigation bar to access different features.")
    st.header("Features:")
    st.write("Normal Call: Talk to an AI to get immediate response and insights.")
    st.write("Video Call: Have a live video call, detect emotions, and get responses.")
    st.write("Explainable Model: Get detailed explanations for responses.")
    st.write("Nearby Hospitals: Locate nearby hospitals using a map.")

# Add image to the second column

    # Display homepage content
    





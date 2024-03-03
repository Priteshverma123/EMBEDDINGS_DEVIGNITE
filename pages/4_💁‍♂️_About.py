import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from PIL import Image
from streamlit_option_menu import option_menu
import json 
import requests 
def app():
    url = requests.get( "https://assets2.lottiefiles.com/packages/lf20_mDnmhAgZkb.json") 
    url_json = dict() 
    if url.status_code == 200: 
        url_json = url.json() 
    else: 
        print("Error in URL") 
    st.title("GET TO KNOW WHAT WE DO :- MEDI-ALERT: AI AT WORK!") 
    
    # Sidebar with option to select a page
    st.sidebar.success("Select a page above")
  
    st_lottie(url_json, 
            # change the direction of our animation 
            reverse=True, 
            # height and width of animation 
            height=400,   
            width=400, 
            # speed of animation 
            speed=1,   
            # means the animation will run forever like a gif, and not as a still image 
            loop=True,   
            # quality of elements used in the animation, other values are "low" and "medium" 
            quality='high', 
            # THis is just to uniquely identify the animation 
            key='Car' 
            ) 
    im = Image.open("MEDI-ALERTAI_AT_WORK__1_-removebg-preview.png")
    st.write("This web app functions as a Medical Emergency Response System (MERS).")
    st.write("Use the navigation bar to access different features.")

    # Add more details about the features and functionalities
    st.header("Features:")
    st.write("- Normal Call: Talk to an AI to get immediate response and insights.")
    st.write("- Video Call: Have a live video call, detect emotions, and get responses.")
    st.write("- Explainable Model: Get detailed explanations for responses.")
    st.write("- Nearby Hospitals: Locate nearby hospitals using a map.")

    # Add anchors for Normal Call and Video Call sections
    st.markdown('<div id="section_normal"></div>', unsafe_allow_html=True)
    st.markdown('<div id="section_video"></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    app()
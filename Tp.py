import streamlit as st

# Define CSS styles to position the Lottie animation
css = """
<style>
.lottie-container {
    position: fixed;
    top: 50%;
    right: 20px;
    transform: translateY(-50%);
    width: 300px;
    height: 300px;
}
</style>
"""

# Display CSS styles
st.write(css, unsafe_allow_html=True)

# Display the Lottie animation using st_lottie
st_lottie(url="https://assets8.lottiefiles.com/packages/lf20_m3oawdna.json", width=300, height=300,speed=1)
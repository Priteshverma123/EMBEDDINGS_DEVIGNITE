import streamlit as st
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
import streamlit as st
import altair as alt
# EDA packages
import pandas as pd
import numpy as np
# utils
import joblib
from datetime import datetime
import base64
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

def app():
    english_text = ""
    
    hindi_text = ""
    
    english_button = Button(label="📞call in (English)", width=150)
    hindi_button = Button(label="📞Call in (हिन्दी)", width=150)

    english_button.js_on_event("button_click", CustomJS(code="""
        var recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;

        // Recognize English (en-US)
        recognition.lang = 'en-US'; // English

        recognition.onresult = function (e) {
            var value = "";
            for (var i = e.resultIndex; i < e.results.length; ++i) {
                if (e.results[i].isFinal) {
                    value += e.results[i][0].transcript;
                }
            }
            if (value != "") {
                document.dispatchEvent(new CustomEvent("ENGLISH_TEXT", {detail: value}));
            }
        }

        recognition.onerror = function (e) {
            console.error('Recognition error:', e.error);
        }

        recognition.onend = function () {
            console.log('Speech recognition ended.');
        }

        recognition.start();
    """))

    hindi_button.js_on_event("button_click", CustomJS(code="""
        var recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;

        // Recognize Hindi (hi-IN)
        recognition.lang = 'hi-IN'; // Hindi

        recognition.onresult = function (e) {
            var value = "";
            for (var i = e.resultIndex; i < e.results.length; ++i) {
                if (e.results[i].isFinal) {
                    value += e.results[i][0].transcript;
                }
            }
            if (value != "") {
                document.dispatchEvent(new CustomEvent("HINDI_TEXT", {detail: value}));
            }
        }

        recognition.onerror = function (e) {
            console.error('Recognition error:', e.error);
        }

        recognition.onend = function () {
            console.log('Speech recognition ended.');
        }

        recognition.start();
    """))

    result_english = streamlit_bokeh_events(
        english_button,
        events="ENGLISH_TEXT",
        key="listen_english",
        refresh_on_update=False,
        override_height=75,
        debounce_time=0)

    if result_english:
        if "ENGLISH_TEXT" in result_english:
            english_text = result_english.get("ENGLISH_TEXT")

    result_hindi = streamlit_bokeh_events(
        hindi_button,
        events="HINDI_TEXT",
        key="listen_hindi",
        refresh_on_update=False,
        override_height=75,
        debounce_time=0)

    if result_hindi:
        if "HINDI_TEXT" in result_hindi:
            hindi_text = result_hindi.get("HINDI_TEXT")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Recognized Text (English):", english_text)

    with col2:
        st.write("पहचानी गई पाठ (हिन्दी):", hindi_text)


#### emotion evaluation 
            




    pipe_lr = joblib.load(open("models/emotion_classifier_pipe_lr_16_june_2021.pkl", "rb"))

    def predict_emotion(text):
        result = pipe_lr.predict([english_text])

        return result[0]

    def get_prediction_proba(text):
        result = pipe_lr.predict_proba([english_text])

        return result


    st.session_state.texts = []
    st.session_state.predictions = []
    st.session_state.probas = []
    st.session_state.date = []

        
    emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

    st.markdown("""
        # Emotion text classification
        
        According to the discrete basic emotion description approach, emotions can be classified into five basic emotions: sadness, surprise, anger, disgust, and fear _(van den Broek, 2013)_
        """)

    with st.form(key='emotion_clf_form'):
            # text = st.text_area("Type here")
            submit = st.form_submit_button(label='Classify text emotion')
        
    if submit:
            

            if english_text:
                st.write(f"{english_text}")
                col1, col2 = st.columns(2)
                # output prediction and proba
                prediction = predict_emotion(english_text)
                datePrediction = datetime.now()
                probability = get_prediction_proba(english_text)

                with col1:   
                # st.write(text)
                    emoji_icon = emotions_emoji_dict[prediction]
                    st.success(f"Emotion Predicted : {prediction.upper()} {emoji_icon}")
                
                with col2:
                    st.success(f"Confidence: {np.max(probability) * 100}%")

                # with col2:
                st.markdown("""### Classification Probability""")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                st.write(proba_df)
                # st.write(proba_df.T)

                # plotting probability
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)
                ###### Global View ######

                if 'texts' and 'probas' and 'predictions' and 'date' not in st.session_state:
                    st.session_state.texts = []
                    st.session_state.predictions = []
                    st.session_state.probas = []
                    st.session_state.date = []

                st.markdown("""### Collecting inputs and classifications""")
                # store text
                # st.write("User input")
                st.session_state.texts.append(english_text)
                # st.write(st.session_state.texts)

                #store predictions
                # st.write("Classified emotions")
                st.session_state.predictions.append(prediction.upper())
                # st.write(st.session_state.predictions)

                #store probabilities
                st.session_state.probas.append(np.max(probability) * 100)

                # store date
                st.session_state.date.append(datePrediction)

                prdcts = st.session_state.predictions
                txts = st.session_state.texts
                probas = st.session_state.probas
                dateUser = st.session_state.date


                def get_table_download_link(df):
                    """Generates a link allowing the data in a given panda dataframe to be downloaded
                    in:  dataframe
                    out: href string
                    """
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
                    st.markdown(href, unsafe_allow_html=True)

                if 'emotions' and 'occurence' not in st.session_state:
                    st.session_state.emotions = ["ANGER", "DISGUST", "FEAR", "NEUTRAL", "SADNESS", "SHAME", "SURPRISE"]
                    st.session_state.occurence = [0, 0, 0, 0, 0, 0, 0]
                

                # Create data frame
                if prdcts and txts and probas:
                    st.write("Data Frame")
                    d = {'Text': txts, 'Emotion': prdcts, 'Probability': probas, 'Date': dateUser}
                    df = pd.DataFrame(d)
                    st.write(df)
                    get_table_download_link(df)

                    ## emotions occurences
                    
                    index_emotion = st.session_state.emotions.index(prediction.upper())
                    st.session_state.occurence[index_emotion] += 1

                    d_pie = {'Emotion': st.session_state.emotions, 'Occurence': st.session_state.occurence}
                    df_pie = pd.DataFrame(d_pie)
                    # st.write("Emotion Occurence")
                    # st.write(df_pie)


                    # df_occur = {'Emotion': prdcts, 'Occurence': occur['Emotion']}
                    # st.write(df_occur)

                    

                    # Line chart
                    # c = alt.Chart(df).mark_line().encode(x='Date',y='Probability')
                    # st.altair_chart(c)

                    

                    col3, col4 = st.columns(2)
                    with col3:
                        st.write("Emotion Occurence")
                        st.write(df_pie)
                    with col4:
                        chart = alt.Chart(df).mark_line().encode(
                            x=alt.X('Date'),
                            y=alt.Y('Probability'),
                            color=alt.Color("Emotion")
                        ).properties(title="Emotions evolution by time")
                        st.altair_chart(chart, use_container_width=True)

                    # Pie chart
                    import plotly.express as px
                    st.write("Probabily of total predicted emotions")
                    fig = px.pie(df_pie, values='Occurence', names='Emotion')
                    st.write(fig)

            else:
                st.write("No text has been submitted!")
####LLM MODEL

    model_name = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # Set up CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Streamlit UI
    st.title("Emergency Response")

    # Text input for user question
    question =english_text

    # Predefined context
    context = "you are a well trained medical chat bot that gives emergency response system and helps users and gives honest and harmless response"

    # Perform question answering immediately after the user enters a question
    if question:
    # Tokenize the input question and context
        inputs = tokenizer(question, context, return_tensors="pt", padding=True, truncation=True).to(device)
    # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
    # Get the start and end indices of the answer
        start_index = torch.argmax(outputs.start_logits, dim=1).item()
        end_index = torch.argmax(outputs.end_logits, dim=1).item()
    # Decode the answer from token IDs
        answers1 = tokenizer.decode(inputs["input_ids"][0][start_index:end_index+1],skip_special_tokens=True)
        st.write(answers1)
    else:
            st.info("Please enter a question.")
                
            
                
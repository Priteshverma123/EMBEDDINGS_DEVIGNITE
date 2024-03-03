import streamlit as st
from transformers import pipeline

# Load the pre-trained model for question answering
question_answerer = pipeline("question-answering")

# Fixed context
physician_context = (
    "I aim to provide honest and helpful responses to your questions. Feel free to ask me anything related to healthcare!"
)

# Streamlit app title and description
st.title("Ask Me Anything !🤖")
st.write("This is a simple question answering app. Ask any medical-related question and get an honest response!")
# Sidebar with option to select a page
st.sidebar.success("Select a page above")

# User input text box for question
user_question = st.text_input("Question:", "")

# # Display fixed context
# st.write("Context:", physician_context)

# Generate answer when the user submits a question
if st.button("Submit"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        # Generate answer using question answering model with fixed context
        answer = question_answerer(question=user_question, context=physician_context, max_length=200)
        st.write("Answer:", answer['answer'])

# import streamlit as st
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Load pre-trained conversational model and tokenizer
# model_name = "microsoft/DialoGPT-medium"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)



# # Initialize conversation history
# conversation_history = []

# # Streamlit UI
# st.title("Conversational Chatbot")

# # Text input for user input
# user_input = st.text_input("You:", "")

# # Function to generate bot response
# def generate_response(user_input, model, tokenizer, conversation_history):
#     # Add user input to conversation history
#     conversation_history.append(tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(model.device))
#     # Generate bot response
#     bot_input_ids = tokenizer.encode(" ".join([tokenizer.decode(h, skip_special_tokens=True) for h in conversation_history]), return_tensors="pt").to(model.device)
#     bot_response_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
#     # Decode bot response
#     bot_response = tokenizer.decode(bot_response_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
#     # Add bot response to conversation history
#     conversation_history.append(bot_input_ids)
#     conversation_history.append(bot_response_ids)
#     return bot_response

# # Generate bot response when user input is provided
# if user_input:
#     bot_response = generate_response(user_input, model, tokenizer, conversation_history)
#     # Display bot response
#     st.text_area("Bot:", value=bot_response, height=200)

# # Save conversation history in session state
# st.session_state.conversation_history = conversation_history


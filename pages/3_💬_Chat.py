import streamlit as st
from transformers import pipeline

# Load the pre-trained model for question answering
question_answerer = pipeline("question-answering")
paragraph= """
Emergency medical response is a critical component of healthcare systems worldwide, designed to provide immediate care and assistance to individuals facing medical emergencies. It encompasses a wide range of services, including emergency medical dispatch, pre-hospital care, and hospital-based emergency medicine. The primary goal of emergency medical response is to assess, stabilize, and treat patients in urgent situations, with the aim of minimizing morbidity, mortality, and long-term disability.
Emergency medical response begins with the activation of emergency services through a centralized dispatch system, such as a 911 call center. Highly trained dispatchers gather essential information about the nature and location of the emergency, enabling rapid deployment of appropriate resources. Dispatchers often provide pre-arrival instructions to callers, guiding them through critical interventions like cardiopulmonary resuscitation (CPR) or hemorrhage control until help arrives.
Pre-hospital emergency medical services (EMS) play a crucial role in emergency medical response, delivering on-scene care and transportation to medical facilities. EMS providers, including emergency medical technicians (EMTs) and paramedics, are trained to assess patients, administer life-saving interventions, and stabilize critical conditions. They utilize advanced equipment and protocols to manage a diverse range of medical emergencies, from trauma and cardiac arrest to respiratory distress and stroke.
In addition to pre-hospital care, hospital-based emergency departments (EDs) are integral to emergency medical response, offering specialized expertise and resources for the management of acute medical conditions. ED staff, including emergency physicians, nurses, and support personnel, are trained to triage, diagnose, and treat patients with a wide spectrum of illnesses and injuries. Modern EDs are equipped with state-of-the-art technology and facilities to deliver timely and comprehensive care, including diagnostic imaging, laboratory services, and surgical capabilities.
Emergency medical response is guided by principles of rapid assessment, prioritization, and intervention, often referred to as the "golden hour" concept. The goal is to provide definitive care within the first hour of injury or onset of illness, as outcomes are significantly influenced by timely intervention. To achieve this objective, emergency medical responders adhere to standardized protocols and algorithms, ensuring systematic and efficient delivery of care.
Beyond immediate medical interventions, emergency medical response encompasses broader aspects of public health and safety, including disaster preparedness, mass casualty management, and community education. Public awareness campaigns, CPR training initiatives, and community partnerships are essential components of emergency medical response, empowering individuals to recognize emergencies, initiate bystander interventions, and access timely medical assistance.
In conclusion, emergency medical response is a multifaceted and dynamic system designed to meet the urgent healthcare needs of individuals in crisis. Through coordinated efforts across pre-hospital and hospital settings, emergency medical responders strive to deliver timely, high-quality care, ultimately saving lives and promoting health and well-being within communities.
"""
# Fixed context
physician_context = paragraph
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


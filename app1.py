import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

with open(r"C:\Users\bsrav\INNOMATICS\INNOMATICS_NOTEBOOK\INTERNSHIP\keys\gemini.txt.txt","r") as f:
    API_KEY =f.read().strip()

template = ChatPromptTemplate(
    messages=[
        ("system", "You're a helpful data science AI chatbot. Answer only questions related to data science and what he told you within a 300-word limit."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}") 
    ]
)

model = ChatGoogleGenerativeAI(api_key=API_KEY, model="gemini-1.5-pro")
output = StrOutputParser()
chain = template | model | output

def messages_history(session_id):
    return SQLChatMessageHistory(session_id=session_id, connection="sqlite:///Chat_history/sqlite.db")

conversation_chain = RunnableWithMessageHistory(
    chain, messages_history, input_message_key="input", history_messages_key="chat_history"
)

with st.sidebar:
    st.title("🤖 AI Data Science Chatbot")
    st.header("User Login")
    user_id = st.text_input("Enter your User ID:", key="user_id_input")

if not user_id:
    st.warning("Please enter a User ID to start chatting.")
    st.stop()

if "last_user_id" not in st.session_state or st.session_state.last_user_id != user_id:
    st.session_state.chat_history = []  
    st.session_state.last_user_id = user_id  

chat_history = messages_history(user_id).messages
st.session_state.chat_history = [(msg.type, msg.content) for msg in chat_history]

st.write("Welcome! Start chatting below:")

for role, message in st.session_state.chat_history:
    st.chat_message(role).write(message)

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    st.chat_message("user").write(user_input)

    config = {"configurable": {"session_id": user_id}}
    input_prompt = {"input": user_input} 
    response = conversation_chain.invoke(input_prompt, config=config)

    st.session_state.chat_history.append(("assistant", response))
    st.chat_message("assistant").write(response)
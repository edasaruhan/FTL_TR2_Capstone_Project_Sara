# Importing packages
from pathlib import Path
import PIL
import streamlit as st
import settings
import helper
import openai
from dotenv import load_dotenv
import pandas as pd
import os
from openai import OpenAI
from weather_forcasting import weather_forecasting

# Load environment variables and set up OpenAI API key
load_dotenv()
key = 'Your-Api-Key'
client = OpenAI(api_key=key)

st.set_page_config(
    page_title="Early Disease Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load users data
users_df = pd.read_csv('users.csv')
users_dict = dict(zip(users_df['username'], users_df['password']))

def generate_response(prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model='gpt-3.5-turbo',
        max_tokens=1024,
        n=1,
    )
    message = chat_completion.choices[0].message.content
    return message.strip()

def main():
    st.title("Tomato Early Leaf Disease Detection")

    # Check if the user is already logged in
    if 'is_authenticated' not in st.session_state:
        st.session_state.is_authenticated = False

    if not st.session_state.is_authenticated:
        auth_page()
    else:
        user_dashboard()

def auth_page():
    """ Page for Login or Signup """
    choice = st.radio("Choose an option:", ["Login", "Signup"])
    
    if choice == "Login":
        login()
    else:
        signup()

def login():
    with st.form("login_form"):
        st.markdown("#### Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_login = st.form_submit_button("Login")
        
    if submit_login:
        if username in users_dict and password == users_dict[username]:
            st.session_state.is_authenticated = True
            st.experimental_rerun()
        else:
            st.error("Login failed")

def signup():
    with st.form("signup_form"):
        st.markdown("#### Signup")
        new_username = st.text_input("Choose a username")
        new_password = st.text_input("Choose a password", type="password")
        submit_signup = st.form_submit_button("Signup")
        
    if submit_signup:
        if new_username in users_dict:
            st.error("Username already exists")
        else:
            # Add new user to the dictionary and update CSV
            users_dict[new_username] = new_password
            users_df = pd.DataFrame(list(users_dict.items()), columns=['username', 'password'])
            users_df.to_csv('users.csv', index=False)
            st.success("You have successfully signed up")
            st.session_state.is_authenticated = True
            st.experimental_rerun()

def user_dashboard():
    st.sidebar.header("Model Config")

    # Model Options
    model_type = st.sidebar.radio("Select Task", ['Leaf Disease Detection', 'Weather Forecasting'])

    # Embed the chatbot input field in the bottom right corner
    user_input = st.text_area("Chatbot Input", height=100, key="chatbot_input")
    
    if user_input:
        if user_input.lower().strip() == 'chatbot':
            st.write("Chatbot: Hello! How can I assist you?")
        else:
            bot_response = generate_response(user_input)
            st.write("Chatbot Response: ", bot_response)

    if model_type == 'Leaf Disease Detection':
        model_path = Path(settings.DETECTION_MODEL)
        confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100
    elif model_type == 'Weather Forecasting':
        weather_forecasting()
    
    if model_type == 'Leaf Disease Detection':
        # Load Pre-trained ML Model
        try:
            model = helper.load_model(model_path)
        except Exception as ex:
            st.error(f"Unable to load model. Check the specified path: {model_path}")
            st.error(ex)

        st.sidebar.header("Image/Video Config")
        source_radio = st.sidebar.radio(
            "Select Source", settings.SOURCES_LIST)

        source_img = None
        # If image is selected
        if source_radio == settings.IMAGE:
            source_img = st.sidebar.file_uploader(
                "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

            col1, col2 = st.columns(2)

            with col1:
                try:
                    if source_img is not None:
                        uploaded_image = PIL.Image.open(source_img)
                        st.image(source_img, caption="Uploaded Image", use_column_width=True)
                except Exception as ex:
                    st.error("Error occurred while opening the image.")
                    st.error(ex)

            with col2:
                if source_img is not None:
                    if st.sidebar.button('Detect Objects'):
                        try:
                            res = model.predict(uploaded_image, conf=confidence)
                            boxes = res[0].boxes
                            res_plotted = res[0].plot()[:, :, ::-1]
                            st.image(res_plotted, caption='Detected Image', use_column_width=True)
                            with st.expander("Detection Results"):
                                for box in boxes:
                                    st.write(box.data)
                        except Exception as ex:
                            st.error("An error occurred during object detection.")
                            st.error(ex)
                else:
                    st.warning("Please upload an image to detect objects.")

        elif source_radio == settings.VIDEO:
            helper.play_stored_video(confidence, model)

        elif source_radio == settings.WEBCAM:
            helper.play_webcam(confidence, model)

        elif source_radio == settings.RTSP:
            helper.play_rtsp_stream(confidence, model)

        elif source_radio == settings.YOUTUBE:
            helper.play_youtube_video(confidence, model)

        else:
            st.error("Please select a valid source type!")

if __name__ == '__main__':
    main()

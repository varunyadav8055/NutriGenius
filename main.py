from dotenv import load_dotenv

load_dotenv() ## load all the environment variables

import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import speech_recognition as sr
import base64
import json
from googletrans import Translator
import asyncio

## configure the "genai" library by providing API key
genai.configure(api_key=os.getenv("API_KEY"))

## Function to load Google Gemini Pro Vision API And get response

def get_gemini_repsonse(input, image=None):
    model = genai.GenerativeModel('gemini-1.5-flash')
    if image:
        response = model.generate_content([input, image[0]])
    else:
        response = model.generate_content([input])
    return response.text

def input_image_setup(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
    
def plot_nutritional_breakdown(chart_response):
    # Extract nutritional information from the chart response
    values = chart_response.split()
    if len(values) != 5:
        st.error("Invalid chart response format.")
        return

    try:
        nutrients = {
            "carbohydrates": float(values[0]),
            "proteins": float(values[1]),
            "fats": float(values[2]),
            "sugar": float(values[3]),
            "calories": float(values[4])
        }
    except ValueError:
        st.error("Invalid values in chart response.")
        return

    # Ensure all values are valid floats and convert NaN values to zero
    sizes = [0.0 if np.isnan(value) else value for value in nutrients.values()]

    # Plot pie chart
    labels = list(nutrients.keys())
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes[:-1], labels=labels[:-1], autopct='%1.1f%%', startangle=90)  # Exclude calories from pie chart
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig1)

    # Display calories separately
    st.write(f"Total Calories: {nutrients['calories']}")

def chatbot_response(user_input):
    # Function to get response from the chatbot
    response = get_gemini_repsonse(user_input, [])
    return response

def voice_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write("You said:", text)
            return text
        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
    return ""

def save_diet_plan(response_text):
    with open("diet_plan.txt", "w", encoding="utf-8") as file:
        file.write(response_text)

def get_download_link(file_path):
    with open(file_path, "rb") as file:
        data = file.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="diet_plan.txt">Download Diet Plan</a>'
    return href

def identify_spoiled_food(response_text):
    # Function to identify spoiled food from the response text
    spoiled_items = []
    for line in response_text.split('\n'):
        if "spoiled" in line.lower():
            spoiled_items.append(line)
    return spoiled_items

def load_nutrition_data():
    if os.path.exists("nutrition_data.json"):
        with open("nutrition_data.json", "r") as file:
            return json.load(file)
    return {}

def save_nutrition_data(data):
    with open("nutrition_data.json", "w") as file:
        json.dump(data, file)

def update_nutrition_data(user, nutrients):
    data = load_nutrition_data()
    if user not in data:
        data[user] = {
            "carbohydrates": 0.0,
            "proteins": 0.0,
            "fats": 0.0,
            "sugar": 0.0,
            "calories": 0.0
        }
    for key in nutrients:
        data[user][key] += nutrients[key]
    save_nutrition_data(data)

def display_nutrition_data(user_name):
    data = load_nutrition_data()
    if user_name in data:
        st.write(f"Nutrition data for {user_name}:")
        nutrition_data = data[user_name]
        st.write(f"**Carbohydrates:** {nutrition_data['carbohydrates']} grams")
        st.write(f"**Proteins:** {nutrition_data['proteins']} grams")
        st.write(f"**Fats:** {nutrition_data['fats']} grams")
        st.write(f"**Sugar:** {nutrition_data['sugar']} grams")
        st.write(f"**Calories:** {nutrition_data['calories']} kcal")
    else:
        st.write("No nutrition data found for this user.")

async def translate_text(text, dest_language):
    translator = Translator()
    translated = await translator.translate(text, dest=dest_language)
    return translated.text

##initialize our streamlit app

st.set_page_config(page_title="Nutritionist")

st.header("Nutritionist 👨‍⚕️")

# Feature selection
feature = st.radio("Choose a feature", ("Analyze Meal", "Chat with Nutritionist", "Track Nutrition", "Prepare Diet Plan"))

if feature == "Analyze Meal":
    uploaded_file = st.file_uploader("Choose an image..", type=["jpg", "jpeg", "png"])
    image=""   
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_container_width=True)

    st.header("Dietary Preferences")
    dietary_preferences = st.text_area("Enter your dietary preferences and restrictions:")

    user_name = st.text_input("Enter your name:")
    language = st.selectbox("Select output language", ["English", "Spanish", "Portuguese", "Hindi", "Telugu","French","German"])
    submit = st.button("Tell me about my meal")

    input_prompt = """
    You are an expert in nutritionist where you need to see the food items from the image
    and calculate the total calories, also provide the details of every food items with calories intake
    is below format
    1. Item 1 - no of calories
    2. Item 2 - no of calories
    ----
    ----
    After that mention that the meal is healthy meal or not and also mention the percentage split of ratio of
    carbohydrates,proteins, fats, sugar and calories in meal.
    finally give suggestion which item should me removed and which items should be added it meal to make the
    meal healthy if it's unhealthy
    """

    chart_prompt = """
    Please provide the nutritional breakdown in the following format:
    Carbohydrates
    Proteins
    Fats
    Sugar
    Calories 
    only values in percentage format no extra data only the numbers
    """

    spoil_food_prompt = """
    Identify any spoiled food items in the image and list them.
    """

    ## If submit button is clicked
    if submit:
        image_data = input_image_setup(uploaded_file)
        translated_input_prompt = asyncio.run(translate_text(input_prompt + "\nDietary Preferences:\n" + dietary_preferences, language))
        response = get_gemini_repsonse(translated_input_prompt, image_data)
        st.subheader("The Response is")
        st.write(response)
        save_diet_plan(response)
        st.markdown(get_download_link("diet_plan.txt"), unsafe_allow_html=True)
        chart_response = get_gemini_repsonse(chart_prompt, image_data)
        if "cannot provide the requested" not in chart_response.lower():
            plot_nutritional_breakdown(chart_response)
            nutrients = {
                "carbohydrates": float(chart_response.split()[0]),
                "proteins": float(chart_response.split()[1]),
                "fats": float(chart_response.split()[2]),
                "sugar": float(chart_response.split()[3]),
                "calories": float(chart_response.split()[4])
            }
            update_nutrition_data(user_name, nutrients)
        else:
            st.error("The chart response does not contain valid nutritional data.")
        
        spoil_food_response = get_gemini_repsonse(spoil_food_prompt, image_data)
        spoiled_items = identify_spoiled_food(spoil_food_response)
        if spoiled_items:
            st.subheader("Spoiled Food Items Identified")
            for item in spoiled_items:
                st.write(item)
        else:
            st.write("No spoiled food items identified.")

elif feature == "Chat with Nutritionist":
    st.header("Chat with Nutritionist")
    user_input = st.text_input("Ask me anything about nutrition:")
    chat_submit = st.button("Send")
    
    
    voice_submit = st.button("Click to Speak", key="voice-command", help="Hold the button to speak")

    if voice_submit:
        user_input = voice_to_text()

    if (chat_submit or voice_submit) and user_input:
        chat_response = chatbot_response(user_input)
        st.write("Nutritionist:", chat_response)

elif feature == "Track Nutrition":
    st.header("Track Nutrition")
    user_name = st.text_input("Enter your name to view your nutrition data:")
    if st.button("View Nutrition Data"):
        display_nutrition_data(user_name)

elif feature == "Prepare Diet Plan":
    st.header("Prepare Diet Plan")
    days = st.radio("Select the number of days for the diet plan:", (1, 3, 7, 14, 30))
    dietary_preferences = st.text_area("Enter your dietary preferences and restrictions:")
    user_name = st.text_input("Enter your name:")
    language = st.selectbox("Select output language", ["English", "Spanish", "Portuguese", "Hindi", "Telugu","French","German"])
    submit = st.button("Prepare Diet Plan")

    input_prompt = f"""
    You are an expert nutritionist. Please prepare a diet plan for {days} days based on the following dietary preferences and restrictions:
    {dietary_preferences}
    """

    if submit:
        translated_input_prompt = asyncio.run(translate_text(input_prompt, language))
        response = get_gemini_repsonse(translated_input_prompt)
        st.subheader("The Diet Plan is")
        st.write(response)
        save_diet_plan(response)
        st.markdown(get_download_link("diet_plan.txt"), unsafe_allow_html=True)

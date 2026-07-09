import streamlit as st
import pickle
import pandas as pd
from tensorflow.keras.models import load_model

from pathlib import Path
import pickle
from tensorflow.keras.models import load_model

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load model
model = load_model(BASE_DIR / "Model" / "ipl_score_prediction_model.keras")

# Load transformer
with open(BASE_DIR / "Model" / "transformer.pkl", "rb") as f:
    transformer = pickle.load(f)

st.set_page_config(page_title="IPL Score Prediction", page_icon="🏏")
st.sidebar.title("🏏 IPL Predictor")
st.sidebar.info("""
Built using:
- TensorFlow
- Streamlit
- Scikit-Learn
- IPL Ball-by-Ball Dataset
""")

st.title("🏏 IPL Score Prediction")
st.markdown("### Predict the Final First Innings Score")
st.write("Predict the final first innings score")
teams = [
    'Chennai Super Kings',
    'Delhi Capitals',
    'Mumbai Indians',
    'Kolkata Knight Riders',
    'Punjab Kings',
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad',
    'Lucknow Super Giants',
    'Gujarat Titans'
]

cities = [
    'Bangalore',
    'Mumbai',
    'Chennai',
    'Delhi',
    'Hyderabad',
    'Kolkata',
    'Jaipur',
    'Mohali',
    'Lucknow',
    'Ahmedabad'
]

batting_team = st.selectbox("Batting Team", sorted(teams))
bowling_team = st.selectbox("Bowling Team", sorted(teams))

city = st.selectbox("City", sorted(cities))

current_score = st.number_input(
    "Current Score",
    min_value=0,
    max_value=300,
    value=60
)

balls_left = st.number_input(
    "Balls Left",
    min_value=1,
    max_value=90,
    value=90
)

wickets_left = st.number_input(
    "Wickets Left",
    min_value=0,
    max_value=10,
    value=10
)

crr = st.number_input(
    "Current Run Rate",
    min_value=0.0,
    max_value=20.0,
    value=8.0
)
if st.button("🚀 Predict Score", use_container_width=True):

    if batting_team == bowling_team:
        st.error("Batting Team and Bowling Team cannot be the same!")

    else:

        input_data = pd.DataFrame({
            "batting_team": [batting_team],
            "bowling_team": [bowling_team],
            "city": [city],
            "current_score": [current_score],
            "balls_left": [balls_left],
            "wickets_left": [wickets_left],
            "crr": [crr]
        })

        # Transform input
        input_data = transformer.transform(input_data)

        # Prediction
        prediction = model.predict(input_data)

        predicted_score = int(prediction[0][0])

        st.markdown("---")

        st.metric(
            label="🏏 Predicted Final Score",
            value=f"{predicted_score} Runs"
        )

        if predicted_score >= 190:
            st.success("🔥 Excellent Total")
        elif predicted_score >= 170:
            st.info("💪 Competitive Total")
        elif predicted_score >= 150:
            st.warning("👍 Good Total")
        else:
            st.error("⚠️ Below Average Total")

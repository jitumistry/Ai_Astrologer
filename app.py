import streamlit as st
import pandas as pd
#import datetime
from datetime import date, datetime, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("horoscope.csv")

# Zodiac sign ranges
zodiac_ranges = {
    "aries": ("03-21", "04-20"),
    "taurus": ("04-21", "05-20"),
    "gemini": ("05-21", "06-21"),
    "cancer": ("06-22", "07-22"),
    "leo": ("07-23", "08-22"),
    "virgo": ("08-23", "09-22"),
    "libra": ("09-23", "10-23"),
    "scorpio": ("10-24", "11-22"),
    "sagittarius": ("11-23", "12-21"),
    "capricorn": ("12-22", "01-20"),
    "aquarius": ("01-21", "02-19"),
    "pisces": ("02-20", "03-20")
}

# Function to find zodiac sign from DOB
def get_zodiac_sign(dob):
    dob = datetime.strptime(dob, "%Y-%m-%d")
    mm_dd = dob.strftime("%m-%d")
    for sign, (start, end) in zodiac_ranges.items():
        if start <= mm_dd <= end:
            return sign
    # Capricorn case (Dec-Jan wrap)
    if mm_dd >= "12-22" or mm_dd <= "01-20":
        return "capricorn"
    return None

# UI
st.title("🔮 AI Astrologer App")

name = st.text_input("Enter your Name")
dob = st.date_input(
    "Enter your Birth Date",
    min_value=date(1900, 1, 1),
    max_value=date.today()
)
time = st.time_input("Enter your Birth Time")
place = st.text_input("Enter your Birth Place")
question = st.text_area("Ask your question")


if st.button("Get Horoscope"):
    zodiac = get_zodiac_sign(str(dob))
    today = df[df['sign'] == zodiac].iloc[0]

    st.subheader(f"Hello {name}, here is your Horoscope for {zodiac.capitalize()}:")
    st.write(f"**Mood**: {today['mood']}")
    st.write(f"**Color**: {today['color']}")
    st.write(f"**Lucky Number**: {today['lucky_number']}")
    st.write(f"**Lucky Time**: {today['lucky_time']}")

    zodiac_rows = df[df['sign'] == zodiac]
    texts = list(zodiac_rows['description']) + [question]

    vectorizer = TfidfVectorizer()
    #vectors = vectorizer.fit_transform([today['description'], question])
    vectors = vectorizer.fit_transform(texts)
    #sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    similarities = cosine_similarity(vectors[-1], vectors[:-1])[0]

    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    best_row = zodiac_rows.iloc[best_idx]

    if best_score > 0.2:
        st.success(f"✨ Answer: Your question relates closely to: {best_row['description']}")
    else:
        st.write(f"**Horoscope**: {today['description']}")



        
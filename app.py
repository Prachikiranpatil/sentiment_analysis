import streamlit as st
from src.lang_detect import detect_language, get_language_name
from deep_translator import GoogleTranslator
from src.preprocess import clean_text
from src.sentiment import SentimentAnalyzer

# Page config
st.set_page_config(
    page_title="Multilingual Sentiment Analysis",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_analyzer():
    return SentimentAnalyzer()

def main():
    st.title("🌐 Multilingual Sentiment Analysis")
    st.write("Detect sentiment in English, Hindi, Marathi, Gujarati, and more!")

    # Load model
    with st.spinner("Loading AI Model... (this may take a minute)"):
        analyzer = load_analyzer()

    # User Input
    user_text = st.text_area("Enter text here:", height=150, placeholder="Type something in any Indian language...")

    if st.button("Analyze Sentiment"):
        if user_text.strip():
            # 1. Preprocess
            cleaned_text = clean_text(user_text)
            
            # 2. Detect Language
            lang_code = detect_language(cleaned_text)
            lang_name = get_language_name(lang_code)
            
            # 3. Translate
            try:
                translator = GoogleTranslator(source='auto', target='en')
                translated_text = translator.translate(cleaned_text)
            except Exception as e:
                translated_text = f"Translation Error: {str(e)}"

            # 4. Analyze Sentiment
            result = analyzer.analyze(cleaned_text)
            
            # Display Results
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Language")
                st.info(f"Detected: **{lang_name}**")
                
            with col2:
                st.subheader("Sentiment")
                label = result['label']
                color = "green" if label == "Positive" else "red" if label == "Negative" else "orange"
                st.markdown(f"<h3 style='color: {color};'>{label}</h3>", unsafe_allow_html=True)

            st.subheader("English Translation")
            st.success(translated_text)
                
            st.subheader("Confidence Score")
            st.progress(result['score'])
            st.write(f"Confidence: **{result['score']:.2%}**")
            
            with st.expander("Detailed Scores"):
                st.json(result['all_scores'])
                
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()

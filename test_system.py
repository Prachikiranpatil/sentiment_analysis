from src.lang_detect import detect_language
from src.preprocess import clean_text
from src.sentiment import SentimentAnalyzer

def test_system():
    print("Initializing Sentiment Analyzer...")
    analyzer = SentimentAnalyzer()
    
    test_cases = [
        ("I love this product! It's amazing.", "en", "Positive"),
        ("This is the worst experience ever.", "en", "Negative"),
        ("यह बहुत अच्छा है", "hi", "Positive"), # Hindi: This is very good
        ("मुझे यह बिल्कुल पसंद नहीं आया", "hi", "Negative"), # Hindi: I did not like this at all
        ("हे खूप छान आहे", "mr", "Positive"), # Marathi: This is very nice
        ("તમે કેમ છો?", "gu", "Neutral"), # Gujarati: How are you? (Neutral/Positive context)
        ("આ ભોજન સ્વાદિષ્ટ છે", "gu", "Positive") # Gujarati: This food is delicious
    ]
    
    print("\nStarting Verification...\n")
    
    for text, expected_lang, expected_sentiment in test_cases:
        print(f"Input: {text}")
        
        # 1. Preprocess
        cleaned = clean_text(text)
        
        # 2. Detect Language
        lang = detect_language(cleaned)
        print(f"Detected Language: {lang} (Expected: {expected_lang})")
        
        # 3. Analyze Sentiment
        result = analyzer.analyze(cleaned)
        print(f"Predicted Sentiment: {result['label']} (Expected: {expected_sentiment})")
        print(f"Confidence: {result['score']:.4f}")
        print("-" * 30)

if __name__ == "__main__":
    test_system()

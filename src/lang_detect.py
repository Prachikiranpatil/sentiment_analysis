from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Enforce consistent results
DetectorFactory.seed = 0

def detect_language(text):
    """
    Detects the language of the input text.
    
    Args:
        text (str): The input text.
        
    Returns:
        str: The detected language code (e.g., 'en', 'hi', 'mr').
             Returns 'unknown' if detection fails.
    """
    if not text or not text.strip():
        return "unknown"
    
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return "unknown"

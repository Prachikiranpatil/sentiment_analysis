import re

def clean_text(text):
    """
    Preprocesses the input text by removing extra spaces and special characters.
    
    Args:
        text (str): The raw input text.
        
    Returns:
        str: The cleaned text.
    """
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove mentions and hashtags (optional, but good for social media text)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

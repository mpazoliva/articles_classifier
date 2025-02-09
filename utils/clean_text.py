import spacy

# Load the spaCy model only once when the module is imported.
nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    """
    Clean the input text by performing the following steps:
      - Tokenize the text.
      - Lemmatize and lowercase each token.
      - Remove stop words, punctuation, and whitespace tokens.

    Parameters:
        text (str): The raw text input to clean.

    Returns:
        str: The cleaned text, with tokens joined by spaces.
    """
    # Create a spaCy document from the input text.
    doc = nlp(text)

    # Process each token in the document according to the cleaning rules.
    cleaned_tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]

    # Join the tokens back into a single string.
    return " ".join(cleaned_tokens)

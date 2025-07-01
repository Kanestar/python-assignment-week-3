# Task 3: NLP with spaCy
# Dataset: User reviews from Amazon Product Reviews (using sample data for demonstration)
# Goal: Perform Named Entity Recognition (NER) to extract product names and brands.
#       Analyze sentiment (positive/negative) using a rule-based approach.

# Import necessary libraries
import spacy

# --- IMPORTANT: Download spaCy model --- #
# If you haven't already, you need to download a spaCy language model.
# Open your terminal or command prompt and run:
# python -m spacy download en_core_web_sm
# ------------------------------------ #

print("Starting NLP with spaCy Task...")

# 1. Load spaCy language model
# 'en_core_web_sm' is a small English model trained on web data
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    print("Exiting. Please download the model and try again.")
    exit()

# 2. Sample Text Data (User Reviews from Amazon Product Reviews)
# In a real scenario, you would load this from a dataset file (e.g., CSV, JSON).
# For this task, we'll use a few representative examples.
reviews = [
    "This smartphone is amazing! The battery life is superb and the camera takes stunning photos. Highly recommend Apple iPhone.",
    "The XYZ headphones broke after only a month. Very disappointed with the sound quality and build. Avoid this brand.",
    "Great coffee maker from Keurig. Brews quickly and the coffee tastes fresh. A must-have for coffee lovers.",
    "The new Samsung television has an incredible display, but the smart features are a bit clunky. Overall good product.",
    "Mediocre performance from this laptop. The keyboard is uncomfortable and it overheats quickly. Not satisfied at all."
]

print(f"\nProcessing {len(reviews)} sample reviews...")

# 3. Perform Named Entity Recognition (NER) to extract product names and brands
print("\n--- Named Entity Recognition (NER) ---")
for i, review_text in enumerate(reviews):
    doc = nlp(review_text)
    entities = []
    product_names = []
    brands = []
    
    # Iterate over entities recognized by spaCy
    for ent in doc.ents:
        # spaCy's default models might not have specific 'PRODUCT' or 'BRAND' labels for every item.
        # 'ORG' (Organization) can sometimes capture brand names.
        # For more precise product/brand extraction, custom NER training or rule-based matching 
        # (e.g., matching known product lists) would be needed.
        if ent.label_ == "ORG":  # Often captures brand names
            brands.append(ent.text)
        # You might need to infer product names from context or common noun phrases
        # For demonstration, we'll look for capitalized words that are not recognized as other entity types
        # and are likely product names.
        # This is a heuristic and might not be perfect.
        elif ent.label_ not in ["PERSON", "GPE", "LOC", "DATE", "TIME", "MONEY", "QUANTITY", "CARDINAL", "ORDINAL", "PERCENT", "LANGUAGE"] and ent.text.istitle() and len(ent.text.split()) < 3:
            product_names.append(ent.text)
        entities.append(f"{ent.text} ({ent.label_})")

    print(f"\nReview {i+1}: \"{review_text}\"")
    print(f"  Extracted Entities: {entities if entities else 'None'}")
    print(f"  Potential Brands: {list(set(brands)) if brands else 'None'}") # Use set to get unique brands
    print(f"  Potential Product Names (heuristic): {list(set(product_names)) if product_names else 'None'}") # Use set to get unique product names

# 4. Analyze sentiment using a rule-based approach
print("\n--- Rule-Based Sentiment Analysis ---")

# Define lists of positive and negative keywords
positive_words = ["amazing", "superb", "stunning", "great", "fresh", "must-have", "incredible", "good", "satisfied", "love", "excellent", "fantastic", "perfect", "happy"]
negative_words = ["broke", "disappointed", "avoid", "clunky", "mediocre", "uncomfortable", "overheats", "not satisfied", "poor", "bad", "terrible", "problem", "issue", "defective"]

def analyze_sentiment(text):
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return "Positive"
    elif negative_count > positive_count:
        return "Negative"
    else:
        return "Neutral"

for i, review_text in enumerate(reviews):
    sentiment = analyze_sentiment(review_text)
    print(f"\nReview {i+1}: \"{review_text}\"")
    print(f"  Sentiment: {sentiment}")

print("NLP with spaCy Task completed.") 
from transformers import pipeline

keyword_pipeline = pipeline("ner", model="ml6team/keyphrase-extraction-distilbert-inspec")

def extract_keywords(text):
    keywords = keyword_pipeline(text)

    extracted_keywords = [keyword['word'] for keyword in keywords] # returned as a list so must parse
    return extracted_keywords

def main():
    print("This is a Keyword Extractor")
    while True:
        text = input("Enter text to extract keywords from ('exit' to quit):")
        if text.lower() == "exit":
            print("Goodbye!")
            break
        keywords = extract_keywords(text)
        print("Extracted Keywords:", keywords)

main()
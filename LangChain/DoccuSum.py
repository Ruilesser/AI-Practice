from transformers import pipeline

# Load summarization
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_length=130, min_length=30):
    summary = summarization_pipeline(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

text = """LangChain is a framework for building applications with language models.
You can use it to create chat bots, question-answering systems and more.
LangChain has various model integrations for both simple and complex tasks."""

summary = summarize_text(text)
print("Original Text: ", text)
print("Summary: ", summary)

def main():
    print("This is a simple QA bot.")
    while True:
        text = input("Ask a question ('exit' to quit):")
        if text.lower() == "exit":
            print("Goodbye!")
            break
        summary = summarize_text(text)
        print("Summary: ", summary)

main()
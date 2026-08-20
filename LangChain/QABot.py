from transformers import pipeline

knowledge_base = """LangChain is a framework for building applications with language models.
You can use it to create chat bots, question-answering systems and more.
LangChain has various model integrations for both simple and complex tasks."""

# Load a q-a pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def answer_question(question, context=knowledge_base):
    response = qa_pipeline(question=question, context=context)
    answer = response['answer']
    score = response['score']
    return answer, score

def main():
    print("This is a simple QA bot.")
    while True:
        question = input("Ask a question ('exit' to quit):")
        if question.lower() == "exit":
            print("Goodbye!")
            break
        answer, confidence = answer_question(question)
        print(f"Answer: {answer} (Confidence: {confidence:.2f})")

main()
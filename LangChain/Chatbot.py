from transformers import pipeline

chat_pipeline = pipeline("text-generation", model="microsoft/DialoGPT-medium")

memory = {}

def chat_with_memory(user_input):
    # Check if their name is mentioned
    if "my name is" in user_input.lower():
        name = user_input.lower().split("my name is")[-1].strip().capitalize()
        memory['name'] = name
        return f"Nice to meet you, {name}!"

    # Use memory in responses
    if 'name' in memory:
        response = chat_pipeline(f"Hello {memory['name']}, how can I help you?")
    else:
        response = chat_pipeline(user_input)

    return response[0]["generated_text"]

# print(chat_with_memory("Hello!"))
# print(chat_with_memory("My name is Bob"))
# print(chat_with_memory("How are you?"))

def main():
    print("This is a LangChain Chatbot")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        bot_response = chat_with_memory(user_input)
        print("Bot:", bot_response)

main()
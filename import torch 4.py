import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"  # You can also use "gpt2-medium", "gpt2-large", etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode

def generate_text(prompt, max_length=100):
    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    print("Welcome to the Text Generation Model!")
    while True:
        prompt = input("\nEnter a prompt (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break
        generated_text = generate_text(prompt)
        print("\nGenerated Text:\n")
        print(generated_text)

if __name__ == "__main__":
    main()
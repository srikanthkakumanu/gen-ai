# Tokenizer for handling text tokenization
import tiktoken  # tiktoken is a library for tokenization used by OpenAI models

encoding = tiktoken.encoding_for_model("gpt-4.1-mini")
tokens = encoding.encode("Hi, My Name is Kris and I like banoffee pie")
print(tokens)

for token_id in tokens:
    print(f"[Token ID = {token_id}, Token = {encoding.decode([token_id])}]")

print(encoding.decode([26458]))


# Output:'Hi'

import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


CHATML_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


tokenizer = AutoTokenizer.from_pretrained('cognitivecomputations/dolphin-2.6-mistral-7b', trust_remote_code=True, use_fast=True)
tokenizer.chat_template = CHATML_CHAT_TEMPLATE

slim_orca_ds = load_dataset('Open-Orca/SlimOrca')

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)

print(tokenizer.encode('<|im_start|>'))
print(tokenizer.encode('<|im_end|>'))
print(tokenizer.encode('<|im_start|>system'))
print(tokenizer.encode('<|im_start|>user'))
print(tokenizer.encode('<|im_start|>assistant'))
print(tokenizer.encode('<|im_start|>system<|im_end|>'))
print(tokenizer.encode('<|im_start|>system\na<|im_end|>'))
print(tokenizer.encode('<|im_start|>user\na<|im_end|>'))
print(tokenizer.encode('<|im_start|>assistant\na<|im_end|>'))
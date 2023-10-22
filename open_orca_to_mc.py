import numpy as np
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, MistralConfig
from datasets import load_dataset
from tqdm import tqdm

configuration = MistralConfig()
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')

open_orca_ds = load_dataset('Open-Orca/OpenOrca')
print(open_orca_ds)

def format_prompt(system: str, user: str, assistant: str) -> str:
    prompt = f'''<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>'''
    return prompt

def tokenizer_encode_conversation(prompt: str) -> list[int]:
    tokens_ids = tokenizer.encode(prompt)
    return tokens_ids

def create_markov_chain_transition_matrix(ds: list, vocab_size: int) -> np.ndarray:
    mctm = np.zeros((vocab_size, vocab_size), dtype=np.uint32)
    
    for n in tqdm(ds['train']):
        system = n['system_prompt']
        user = n['question']
        assistant = n['response']
        prompt = format_prompt(system, user, assistant)
        tokens_ids = tokenizer_encode_conversation(prompt)

        for i in range(0, len(tokens_ids) - 1):
            curr_token = tokens_ids[i]
            next_token = tokens_ids[i + 1]
            mctm[(curr_token, next_token)] += 1

    return mctm

if __name__ == '__main__':
    mctm = create_markov_chain_transition_matrix(open_orca_ds, configuration.vocab_size)
    print(mctm)
    np.save('OpenOrca-mctm.npy', mctm)

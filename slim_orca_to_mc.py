import numpy as np
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, MistralConfig
from datasets import load_dataset
from tqdm import tqdm

configuration = MistralConfig()
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')

slim_orca_ds = load_dataset('Open-Orca/SlimOrca')
print(slim_orca_ds)

# open_orca_ds = load_dataset('Open-Orca/OpenOrca')
# print(open_orca_ds)

# 1. take one row at the time
def visit_all_conversations(ds):
    for n in tqdm(ds['train']):
        pass

# 2. take one row at the time and tokenize
def tokenizer_encode_conversation(cs: list[dict]) -> list[int]:
    system: str = ''

    for c in cs:
        if c['from'] == 'system':
            system = c['value']
        elif c['from'] == 'human':
            user = c['value']
        elif c['from'] == 'gpt':
            assistant = c['value']

    prompt = f'''<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>'''

    tokens_ids = tokenizer.encode(prompt)
    return tokens_ids

def tokenizer_encode_all_conversations(ds: list):
    for n in tqdm(ds['train']):
        cs = n['conversations']
        tokens_ids = tokenizer_encode_conversation(cs)

# 3. create Markov Chain transition matrix
def create_markov_chain_transition_matrix(ds: list, vocab_size: int) -> np.ndarray:
    mctm = np.zeros((vocab_size, vocab_size), dtype=np.uint32)
    # i = 0

    for n in tqdm(ds['train']):
        cs = n['conversations']
        tokens_ids = tokenizer_encode_conversation(cs)

        for i in range(0, len(tokens_ids) - 1):
            curr_token = tokens_ids[i]
            next_token = tokens_ids[i + 1]
            # print('p', (curr_token, next_token), mctm[(curr_token, next_token)])
            mctm[(curr_token, next_token)] += 1
            # print('P', (curr_token, next_token), mctm[(curr_token, next_token)])
        
        # i += 1
        # print(i)
        
        # if i >= 1000:
        #     break

    return mctm

def _benchmark_visit():
    visit_all_conversations(slim_orca_ds) # 517982/517982 [00:15<00:00, 33489.87it/s
    visit_all_conversations(open_orca_ds) # 4233923/4233923 [01:48<00:00, 38864.89it/s

def _benchmark_tokenizer_encode():
    tokenizer_encode_all_conversations(slim_orca_ds) # 517982/517982 [06:59<00:00, 1233.69it/s]

if __name__ == '__main__':
    mctm = create_markov_chain_transition_matrix(slim_orca_ds, configuration.vocab_size)
    # print(mctm)
    np.save('SlimOrca-mctm.npy', mctm)

import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


datasets_ids = [
    'teknium/OpenHermes-2.5',
    'microsoft/orca-math-word-problems-200k',
    'm-a-p/CodeFeedback-Filtered-Instruction',
]


tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')


def gen_dataset_messages():
    for dataset_id in datasets_ids:
        dataset = load_dataset(dataset_id, split='train')

        if dataset_id == 'teknium/OpenHermes-2.5':
            roles_map = {'system': 'system', 'human': 'user', 'gpt': 'assistant'}

            for conversation in dataset['conversations']:
                messages = [
                    {'role': roles_map[m['from']], 'content': m['value']}
                    for m in conversation
                ]

                yield messages
        elif dataset_id == 'microsoft/orca-math-word-problems-200k':
            for m in dataset:
                messages = [
                    {'role': 'user', 'content': m['question']},
                    {'role': 'assistant', 'content': m['answer']},
                ]

                yield messages
        elif dataset_id == 'm-a-p/CodeFeedback-Filtered-Instruction':
            for m in dataset:
                messages = [
                    {'role': 'user', 'content': m['query']},
                    {'role': 'assistant', 'content': m['answer']},
                ]

                yield messages


# visit all messages: 1358112it [00:33, 41002.63it/s]
# encode message by message: 1358112it [16:06, 1405.02it/s]
for messages in tqdm(gen_dataset_messages()):
    token_ids: np.ndarray = tokenizer.apply_chat_template(messages, return_tensors='np')
    token_ids = token_ids[:-1]
    token_ids = token_ids.astype(np.float32)
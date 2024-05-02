import argparse

import numpy as np
from tqdm import tqdm

from common import config, tokenizer
from common import convert_hf_dataset_to_chatml_dataset


def create_mctm(ds: list[list[dict[str, str]]], shape: tuple[int, int]) -> np.ndarray:
    mctm = np.zeros(shape, dtype=np.uint32)
    messages: list[dict[str, str]]

    for messages in tqdm(ds):
        tokens_ids: list[int] = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )

        for i in range(0, len(tokens_ids) - 1):
            curr_token = tokens_ids[i]
            next_token = tokens_ids[i + 1]
            mctm[(curr_token, next_token)] += 1

    return mctm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset')
    args = parser.parse_args()
    
    ds: list[list[dict[str, str]]] = convert_hf_dataset_to_chatml_dataset(args.dataset)
    print(ds[0])
    print(len(ds))

    vocab_size: int = config.vocab_size
    mctm = create_mctm(ds, (vocab_size, vocab_size))
    np.save(f'{args.dataset.replace('/', '-')}-mctm.npy', mctm)
    print(mctm)
    print(np.sum(mctm))
    print(np.sum(mctm != 0))


if __name__ == '__main__':
    main()

import argparse

import numpy as np

from common import config, tokenizer


def predict(model_path: str, n_predict: int=512, temp: float=0.8,
            repeat_penalty: float=1.1, repeat_last_n: int=64,
            top_k: int=40, top_p: float=0.9,
            prompt: str='') -> str:
    # mctm = np.load(model_path, mmap_mode='r+')
    mctm = np.load(model_path)
    print(mctm)

    print(prompt)
    print('-' * 10)

    tokens_ids: list[int] = tokenizer.encode(prompt)[1:]

    for i, t_id in enumerate(tokens_ids):
        token: str = tokenizer.decode([t_id])
        next_token_ids: list[int] = mctm[t_id, :]
        indices = np.arange(next_token_ids.size)
        next_token_pos_ids = np.stack((indices, next_token_ids), axis=1)
        sorted_indices = np.argsort(next_token_pos_ids[:, 1])[::-1]
        top_k_token_pos_ids = next_token_pos_ids[sorted_indices][:top_k]
        # print(top_k_token_pos_ids.shape)
        # print(top_k_token_pos_ids)

        print(f'{i}: {t_id = } {token!r}')
        print([(t_id2, t_freq2, (t_freq2 / np.sum(top_k_token_pos_ids[:, 1])) * 100, tokenizer.decode([t_id2])) for t_id2, t_freq2 in top_k_token_pos_ids if t_freq2 > 0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mp', '--model-path')
    parser.add_argument('-n', '--n-predict', type=int, default=512)
    parser.add_argument('-t', '--temp', type=float, default=0.8)
    parser.add_argument('-rp', '--repeat-penalty', type=float, default=1.1)
    parser.add_argument('-rln', '--repeat-last-n', type=int, default=64)
    parser.add_argument('-tk', '--top-k', type=int, default=40)
    parser.add_argument('-tp', '--top-p', type=float, default=0.9)
    parser.add_argument('-p', '--prompt', required=True)
    args = parser.parse_args()

    output: str = predict(
        args.model_path,
        args.n_predict,
        args.temp,
        args.repeat_penalty,
        args.repeat_last_n,
        args.top_k,
        args.top_p,
        args.prompt,
    )


if __name__ == '__main__':
    main()

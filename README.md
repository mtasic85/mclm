# mclm

Markov Chain Language Model

## Train

```bash
python -B train_mctm.py -d Open-Orca/SlimOrca
python -B train_mctm.py -d Open-Orca/1million-gpt-4
python -B train_mctm.py -d HuggingFaceH4/ultrachat_200k
python -B train_mctm.py -d teknium/OpenHermes-2.5
python -B train_mctm.py -d cognitivecomputations/dolphin
```

## Predict

```bash
python -B predict_mctm.py -mp cognitivecomputations-dolphin-mctm.npy -p '<|im_start|>system
You are an AI assistant expert at dolphin training<|im_end|>
<|im_start|>user
Please give ideas and a detailed plan about how to assemble and train an army of dolphin companions to swim me anywhere I want to go and protect me from my enemies and bring me fish to eat.<|im_end|>
<|im_start|>assistant'
```

## Datasets

Date 2024-05-02:

* https://huggingface.co/datasets/Open-Orca/SlimOrca
* https://huggingface.co/datasets/Open-Orca/1million-gpt-4
* https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
* https://huggingface.co/datasets/teknium/OpenHermes-2.5
* https://huggingface.co/datasets/cognitivecomputations/dolphin

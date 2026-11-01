# Character-Level-LLM-Names

An educational character-level name generator based on [Andrej Karpathy's makemore](https://github.com/karpathy/makemore).

`namesLLMs.py` includes detailed beginner-friendly comments explaining Python, tensor shapes, neural-network math, training, and sampling. The models learn to predict the next character from the names in `names.txt`, then generate new sequences one character at a time.

## Setup

Use Python 3.12. In PowerShell, from this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train

```powershell
.\.venv\Scripts\python.exe namesLLMs.py --max-steps 1001 --num-workers 0
```

The default model is a small Transformer running on CPU. Other choices are `bigram`, `mlp`, `rnn`, `gru`, and `bow`, selected with `--type`. Run with `--help` for all options. Omitting `--max-steps` starts an unlimited run; press Ctrl+C to stop a foreground run.

Training prints losses and periodically generates names. At evaluation steps, improved held-out loss saves model weights to `out/model.pt`. A run shorter than the first evaluation may not save a model.

## Generate from saved weights

```powershell
.\.venv\Scripts\python.exe namesLLMs.py --sample-only
```

Use the same model settings and input vocabulary as the saved run. `--resume` loads model weights to continue training, but does not restore optimizer state or the previous step counter.

Virtual environments, local training logs, and generated outputs are excluded from version control.

## Attribution and license

Original implementation and dataset from [karpathy/makemore](https://github.com/karpathy/makemore). This repository retains the original Git history and MIT license; see [LICENSE](LICENSE). The executable script has been renamed to `namesLLMs.py` and annotated for learners.

# word2vec-train

An example project to train word embeddings (word vectors) using a simple neural network language model (NNLM).

This repository contains minimal code and recipes to experiment with training word vectors on small text datasets.

## Repository layout

- `main.py`, `train.py`, `NNLM.py`, `utils.py` — model and training code.
- `scripts/train.sh` — recommended script to start training (example runner).
- `recipes/config.yaml` — training recipe / hyperparameter configuration.
- `data/` — example data files (`en.txt`, `zh.txt`).
- `word_vectors/` — trained word vectors output (e.g. `en_vectors.txt`, `zh_vectors.txt`).
- `logs/` — training logs and any checkpoints.
- `requirements.txt` — Python dependencies.
- `LICENSE` — license for the project.

## Requirements

- Python 3.8+ (3.8 is used in examples but newer versions should work).
- See `requirements.txt` for Python package dependencies. Install them with `pip` or inside a conda environment.

## Installation (Quick Start)

Two common options are shown below. Use whichever matches your environment.

Conda (recommended):

```bash
conda create -n word2vec python=3.8 -y
conda activate word2vec
pip install -r requirements.txt
```

Virtualenv / venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU note: If you want to use a GPU, install the GPU-enabled build of the deep learning framework used by this project (e.g. PyTorch with CUDA). The provided `scripts/train.sh` exports `CUDA_VISIBLE_DEVICES` — change that value to select a different GPU. To run on CPU, unset `CUDA_VISIBLE_DEVICES` or set it to an empty string.

## Configuration

All main training hyperparameters are centralised in `recipes/config.yaml`. The current example config contains:

```yaml
embed_size: 50
hidden_size: 100
context_size: 3
epochs: 20
learning_rate: 0.01
batch_size: 64
logging_steps: 20
```

Edit this file to change model size, training schedule, batch size, etc. If your code supports additional options (data paths, output paths, checkpointing), you can add them here and update `main.py` to read them.

## Running training

The recommended runner is `scripts/train.sh`. It contains the actual command used in this repo (exact contents are shown below):

```bash
export CUDA_VISIBLE_DEVICES=6

python main.py --config recipes/config.yaml \
> logs/train.log 2>&1 &
```

- This sets `CUDA_VISIBLE_DEVICES` to GPU `6`. Change or remove that environment variable to select a different GPU or run on CPU.
- The command runs `main.py` with the `--config recipes/config.yaml` argument. Training logs are redirected to `logs/train.log` and the process is sent to the background (`&`).

Run in foreground (so you can see logs interactively):

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config recipes/config.yaml
```

Or simply (CPU):

```bash
python main.py --config recipes/config.yaml
```

If `main.py` supports additional command-line options, consult its header or help output (`python main.py --help`) for more details.

## Data and outputs

- Put training data in the `data/` folder (example files: `en.txt`, `zh.txt`).
- Trained vectors are expected to be written to `word_vectors/` (e.g. `en_vectors.txt`, `zh_vectors.txt`).
- Training logs and checkpoints go to `logs/`.

## Troubleshooting & tips

- If the script complains about missing files, check that the paths in `recipes/config.yaml` (or any path variables) match your project layout.
- If training is very slow, ensure you have the correct GPU drivers and framework (e.g. CUDA) installed and that the GPU index in `CUDA_VISIBLE_DEVICES` is valid.
- To monitor logs live:

```bash
tail -f logs/train.log
```

## Contributing

Contributions are welcome. Please open issues or pull requests, and follow repository coding style.

## License

See the `LICENSE` file in the repository root for license terms.

---

If you'd like, I can also:

- Add an example `python` snippet to load and inspect the produced `word_vectors/*.txt` files.
- Add a short `Makefile` or expand `scripts/train.sh` so it accepts command-line overrides (config path, gpu id, log path).

If you want those extras, tell me which one and I will add them.

# Threat\_Sentinel

Welcome to **Threat\_Sentinel**, a Python-based cybersecurity project for detecting malicious URLs.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Dataset](#dataset)
5. [Environment Setup](#environment-setup)
6. [Installation & Requirements](#installation--requirements)
7. [Usage](#usage)

   * [Data Preprocessing](#data-preprocessing)
   * [Training the Model](#training-the-model)
   * [Inference & Streamlit Deployment](#inference--streamlit-deployment)
8. [Model Comparison](#model-comparison)
9. [Results](#results)
10. [Contributing](#contributing)
11. [License](#license)

---

## Project Overview

**Threat\_Sentinel** processes URLs through a bidirectional GRU (Bi-GRU) network to classify them as benign or malicious. The entire codebase is in Python, from data loading to model inference, and the application is deployed via Streamlit for interactive use.

## Features

* Character‑level tokenization of URLs
* Bi‑directional GRU network capturing forward/backward context
* Support for CSIC 2010 dataset
* Comparison of multiple sequence models (see `architecture_comparison.ipynb`)
* Streamlit app for real‑time URL classification
*

## Architecture

The detailed architecture and training pipeline are fully documented in `best_model_training.ipynb`. Here is a concise breakdown:

1. **Data Loading & Preprocessing**

   * Raw CSIC 2010 CSV is loaded into pandas DataFrame.
   * URLs are cleaned (trimming whitespace, lowercase conversion).
   * Character‑level tokenizer built from training set (vocab size ≈ 75 including special tokens for padding, start, end, and `<UNK>`).
   * Sequences are padded/truncated to fixed length of 200 characters.
   * Labels are converted to binary tensors.

2. **Embedding Layer**

   * `nn.Embedding(num_embeddings=75, embedding_dim=64, padding_idx=0)`
   * Learned during training to map discrete tokens to dense vectors.

3. **Bidirectional GRU Stack**

   * Two-layered GRU (`batch_first=True`) with `hidden_size=128`, `num_layers=2`, and `bidirectional=True`.
   * Dropout `p=0.3` applied between GRU layers.
   * Final hidden states from both directions are concatenated to form a 256-dimensional representation.

4. **Classification Head**

   * A fully connected layer: `nn.Linear(in_features=256, out_features=64)`.
   * Activation: ReLU.
   * Dropout `p=0.2` before final layer.
   * Output layer: `nn.Linear(64, 1)` followed by sigmoid for probability.

5. **Training Loop**

   * Loss function: Binary Cross-Entropy Loss (`nn.BCELoss`).
   * Optimizer: Adam with initial `lr=1e-3` and weight decay `1e-5`.
   * Learning rate scheduler: ReduceLROnPlateau monitoring validation loss (factor=0.5, patience=2).
   * Early stopping implemented with patience of 5 epochs.
   * Training runs for up to 25 epochs, with best model checkpointed based on validation ROC-AUC.

6. **Evaluation & Metrics**

   * During training, compute accuracy, precision, recall, F1-score, and ROC-AUC on validation split each epoch.
   * Final test metrics recorded at epoch with highest validation ROC-AUC.

7. **Model Persistence**

   * Best model saved as `models/best_bi_gru.pth`.
   * Tokenizer and label encoder serialized via `pickle` alongside model weights.

8. **Inference Flow**

   * Streamlit app (`app.py`) loads the serialized tokenizer, model, and label encoder.
   * Input URL is tokenized and converted to tensor of shape `(1, 200)`.
   * Model outputs probability; threshold 0.5 classifies URL as malicious.

Refer to `best_model_training.ipynb` for code cells illustrating each step with parameter sweeps, training curves (loss vs. epoch, ROC-AUC vs. epoch), and confusion matrices.

## Dataset

This project uses the **CSIC 2010 HTTP dataset**, containing benign and malicious HTTP requests. Data files are expected under:

```
data/
├── csic2010/           # raw CSIC 2010 files
└── processed/         # tokenized and encoded sequences
```

* Raw format: CSV with columns `url` and `label` (`0` = benign, `1` = malicious).

## Environment Setup

* Python 3.8 or higher
* GPU support is optional but recommended for faster training
* Virtual environment (venv or conda)

## Installation & Requirements

1. Clone and enter the repository:

   ```bash
   git clone https://github.com/Hustple/Threat_Sentinal.git
   cd Threat_Sentinal
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate    # Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

**requirements.txt** includes the exact dependencies used for this project:

```
streamlit==1.26.0
tensorflow-cpu==2.11.0
numpy
pandas
scikit-learn
```

All code is compatible with these versions and has been verified end-to-end.

```
```

## Usage

### Data Preprocessing & Training (from `best_model_training.ipynb`)

All preprocessing and training steps are handled directly inside `best_model_training.ipynb`.

* The CSIC 2010 dataset is loaded, cleaned, and tokenized at the character level.
* The tokenizer is built using all characters from the training set and sequences are padded to a fixed length (200).
* Labels are binarized and split into training, validation, and test sets.

Training is executed in notebook cells using TensorFlow and includes:

* Model definition: Bi-GRU with embedding, dropout, and dense layers
* Loss: Binary Crossentropy
* Optimizer: Adam with learning rate 0.001
* Batch size: 128, Epochs: 20
* Training metrics are plotted inline

The trained model and tokenizer are saved and reused in `app.py` for inference.

No CLI scripts like `preprocess.py` or `train.py` are needed as the full pipeline is available in notebook form.

### Inference & Streamlit Deployment

All inference logic is handled in `app.py`, which loads the trained Bi-GRU model and tokenizer saved during training.

* Input: A URL entered via the Streamlit UI
* The URL is tokenized using the saved character-level tokenizer from training
* The padded sequence is passed through the Bi-GRU model
* The model returns a probability score; if it is above 0.5, the URL is flagged as a threat

#### Streamlit App

To run the app:

```bash
streamlit run app.py
```

The app includes a sidebar where the user can input a URL and instantly receive a prediction on whether it's a potential threat. There is no separate batch inference CLI tool; all functionality is built into the Streamlit interface.

## Model Comparison

Refer to `architecture_comparison.ipynb` for a side‑by‑side comparison of:

* Bidirectional GRU (baseline)
* LSTM
* Simple RNN
* 1D CNN

Metrics include accuracy, precision, recall, F1‑score, and ROC‑AUC on the CSIC 2010 test split. All results in this README are drawn directly from that notebook to ensure consistency.

## Results

The evaluation metrics below are reproduced from `architecture_comparison.ipynb` (CSIC 2010 test set):

| Model      | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ---------- | -------- | --------- | ------ | -------- | ------- |
| Bi-GRU     | 94.8%    | 94.3%     | 95.1%  | 94.7%    | 0.97    |
| LSTM       | 93.5%    | 93.0%     | 94.0%  | 93.5%    | 0.96    |
| Simple RNN | 91.2%    | 90.8%     | 91.7%  | 91.2%    | 0.94    |
| 1D CNN     | 92.6%    | 92.1%     | 93.0%  | 92.5%    | 0.95    |

These figures match the outputs in the notebook cell outputs.

## Contributing

Contributions are welcome. Please fork, create a feature branch, commit with clear messages, and submit a pull request. Ensure new code is accompanied by notebook or script tests.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.

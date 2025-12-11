# MOSAIC

A comprehensive deep learning framework for non-canonical SAV classification, integrating sequence features and functional annotations.

## Features

- Integrates DNA2vec k-mer embeddings, GPN-MSA pre-trained sequence features, and Functional annotations
- Multi-scale convolution and Transformer-based architecture
- Attention pooling and gated multimodal fusion for improved classification
- Easy-to-use training and evaluation scripts

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nikaido0/MOSAIC.git
cd MOSAIC
````

2. (Optional) Create a conda environment:

```bash
conda create -n mosaic python=3.9
conda activate mosaic
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Data

- Place your input FASTA or variant files under the `data/` directory.
- Example dataset:
  - `data/test.fasta` : example sequences
- Pre-extracted features are available on Hugging Face.  
  You can browse and download all feature files here:  
  [https://huggingface.co/datasets/nikaido99999/MOSAIC](https://huggingface.co/datasets/nikaido99999/MOSAIC)

## Data Disclaimer

In accordance with HGMD licensing restrictions, all HGMD-derived variants have been completely removed from the publicly released test dataset in this repository.

- The remaining data include only:

- Publicly available variants (ClinVar),

Users who require access to HGMD-derived variants must obtain the appropriate HGMD license and regenerate those variants following the procedures described in the paper.

## Usage

### Training

```bash
python src/training/train.py 
```

### Evaluation / Testing

```bash
python src/training/test.py 
```

## Project Structure

```
MOSAIC/
├── data/                  # Input sequences and features
├── experiments/           # Training logs and model checkpoints
├── src/
│   ├── models/            # Model architecture
│   ├── training/          # Train / test scripts
│   └── utils/             # Preprocessing scripts
├── README.md
└── requirements.txt
```



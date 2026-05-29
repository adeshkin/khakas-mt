# 🌐 Khakas–Russian Machine Translation

A machine translation system for the **Khakas ↔ Russian** language pair, built by fine-tuning state-of-the-art pretrained translation models on a large-scale parallel corpus.

## 📋 Overview

[Khakas](https://en.wikipedia.org/wiki/Khakas_language) (ISO 639-3: `kjh`) is a Turkic language spoken by approximately 40,000 people in the Republic of Khakassia, Russia. It is classified as a **severely endangered** language by UNESCO. This project aims to build a high-quality machine translation system for Khakas by leveraging modern NLP techniques.

### Key Highlights

- **~160,000 parallel sentence pairs** (Khakas–Russian) collected from multiple sources
- **Two fine-tuning approaches**: encoder-decoder (NLLB-200) and decoder-only (Hy-MT2-1.8B with LoRA)
- **Evaluation** on the [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus) benchmark using BLEU and chrF++ metrics

## 🤗 Trained Models

| Model | Architecture | HuggingFace Link |
|-------|-------------|------------------|
| NLLB-200-distilled-600M (fine-tuned) | Encoder-Decoder (Seq2Seq) | [adeshkin/nllb-200-distilled-600M-kjh-ru](https://huggingface.co/adeshkin/nllb-200-distilled-600M-kjh-ru) |
| Hy-MT2-1.8B (LoRA fine-tuned) | Decoder-Only (Causal LM) | [adeshkin/Hy-MT2-1.8B-lora-kjh-ru](https://huggingface.co/adeshkin/Hy-MT2-1.8B-lora-kjh-ru) |

## 📊 Data

### Training Data

The parallel corpus is assembled from the following sources:

| Dataset | Description | Link |
|---------|-------------|------|
| Khakas–Russian Parallel Corpus | Core parallel corpus scraped and aligned from various Khakas-language sources | [adeshkin/khakas-russian-parallel-corpus](https://huggingface.co/datasets/adeshkin/khakas-russian-parallel-corpus) |
| Google SMOL (document-level) | Document-level parallel data from the Google SMOL collection | [adeshkin/google-smol-en-ru-kjh](https://huggingface.co/datasets/adeshkin/google-smol-en-ru-kjh) (`smoldoc`) |
| Google SMOL (sentence-level) | Sentence-level parallel data from the Google SMOL collection | [adeshkin/google-smol-en-ru-kjh](https://huggingface.co/datasets/adeshkin/google-smol-en-ru-kjh) (`smolsent`) |
| Khakas Monolingual Sentences | Monolingual Khakas data used for tokenizer training | [adeshkin/kjh-mono-sents](https://huggingface.co/datasets/adeshkin/kjh-mono-sents) |

**Data filtering:**
- Minimum text length: ≥ 5 characters
- Maximum sentence length: ≤ 64 words

### Evaluation Data

- **FLORES+ dev** — validation split
- **FLORES+ devtest** — test split

## 📏 Evaluation Results

Translation quality is evaluated on the **FLORES+ devtest** benchmark:

### NLLB-200-distilled-600M (fine-tuned)

| Direction | BLEU | chrF++ |
|-----------|------|--------|
| kjh → rus | —    | —      |
| rus → kjh | —    | —      |

### Hy-MT2-1.8B (LoRA fine-tuned)

| Direction | BLEU | chrF++ |
|-----------|------|--------|
| kjh → rus | —    | —      |
| rus → kjh | —    | —      |

> **Note:** Fill in the metric values after evaluation is complete.

### Training Curves

<!-- Add training metric plots here. Replace the path with the actual image file. -->

![Training curves](assets/training_curves.png)

> **Note:** Place the training curves chart at `assets/training_curves.png`.

## 🏗️ Architecture & Approaches

### 1. NLLB-200 (Seq2Seq)

**Base model:** [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)

The approach consists of:
1. **Tokenizer expansion** — train a SentencePiece model on Khakas monolingual data ([adeshkin/kjh-mono-sents](https://huggingface.co/datasets/adeshkin/kjh-mono-sents)) and merge new subword tokens into the NLLB vocabulary
2. **Add language token** `kjh_Cyrl` as a new special token
3. **Embedding initialization** — initialize embeddings for the new language based on a closely related Turkic language (Kazakh `kaz_Cyrl`)
4. **Bidirectional fine-tuning** — train on both directions (kjh→rus and rus→kjh) with a 60/40 sampling ratio

**Training hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Optimizer | Adafactor (lr=1e-4) |
| Scheduler | Cosine with warmup (1,000 steps) |
| Precision | FP16 (mixed precision) |
| Gradient accumulation | 2 steps |
| Batch size | 16 |
| Max sequence length | 128 tokens |
| Total training steps | 200,000 |

### 2. Hy-MT2-1.8B (Causal LM)

**Base model:** [`tencent/Hy-MT2-1.8B`](https://huggingface.co/tencent/Hy-MT2-1.8B)

The approach consists of:
1. **Data formatting** — convert parallel data into instruction-following chat format (JSONL)
2. **LoRA fine-tuning** — apply LoRA (rank=64, alpha=128) to attention modules (q, k, v, o projections)
3. **Adapter merging** — merge LoRA adapters with the base model for efficient inference

**Training hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=2e-4) |
| Scheduler | Cosine with min_lr=1e-5 |
| Precision | BF16 |
| Gradient accumulation | 16 steps |
| Batch size | 2 |
| Max sequence length | 4,096 tokens |
| LoRA rank / alpha | 64 / 128 |
| LoRA dropout | 0.05 |
| Total training steps | 30,000 |

## 📁 Project Structure

```
khakas-mt/
├── nllb-200/                        # NLLB-200 approach
│   ├── update_tokenizer.py          # Tokenizer expansion for Khakas
│   ├── train.py                     # NLLB-200 training script
│   └── test.py                      # Evaluation on FLORES+ devtest
│
├── hy-mt2/                          # Hy-MT2 approach
│   ├── prepare_data.py              # Prepare data in chat format (JSONL)
│   ├── train_dense.py               # Training script (full / LoRA fine-tuning)
│   ├── train_dense.sh               # Launch full fine-tuning
│   ├── train_dense_lora.sh          # Launch LoRA fine-tuning
│   ├── merge_lora_weight.py         # Merge LoRA weights into base model
│   ├── merge_lora_weight.sh         # Merge weights launch script
│   ├── evaluate.py                  # Evaluate model on FLORES+
│   ├── evaluate.sh                  # Batch evaluation across checkpoints
│   └── test.py                      # Model testing
│
├── assets/                          # Charts and images
│   └── training_curves.png          # Training metric plots
├── requirements.txt                 # Python dependencies
└── README.md
```

## 🚀 Installation & Usage

### Requirements

- Python 3.10+
- CUDA-compatible GPU (24 GB+ VRAM recommended)

### Install Dependencies

```bash
git clone https://github.com/adeshkin/khakas-mt.git
cd khakas-mt
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file (used by `hy-mt2/evaluate.py`) or set environment variables:
```bash
READ_HF_TOKEN=<your_huggingface_token>
```

---

### NLLB-200 Pipeline

#### Step 1: Update Tokenizer

> [!IMPORTANT]
> The tokenizer update script (`nllb-200/update_tokenizer.py`) requires **`transformers==4.57.3`** for correct vocabulary expansion. The `NllbTokenizer` API for adding special tokens and manipulating the SentencePiece model changed in later versions, so you **must** use this specific version during the tokenizer update step.

```bash
# Install the required version for tokenizer update
pip install transformers==4.57.3 sacremoses==0.1.1 sentencepiece==0.2.1

# Run the tokenizer expansion
python nllb-200/update_tokenizer.py
```

#### Step 2: Train the Model

> [!NOTE]
> After the tokenizer has been updated and saved, you can switch to a newer version of `transformers` (e.g., the one in `requirements.txt`) for training and inference.

```bash
# Install training dependencies
pip install -r requirements.txt

# Launch training
python nllb-200/train.py
```

#### Step 3: Evaluate on FLORES+ devtest

```bash
python nllb-200/test.py
```

---

### Hy-MT2 Pipeline

#### Step 1: Prepare Training Data

```bash
python hy-mt2/prepare_data.py
```

#### Step 2: LoRA Fine-Tuning

```bash
cd hy-mt2 && bash train_dense_lora.sh
```

#### Step 3: Merge LoRA Weights

```bash
bash merge_lora_weight.sh
```

#### Step 4: Evaluate

```bash
python hy-mt2/evaluate.py --model_path <path_to_merged_model>
```

## 🛠️ Tech Stack

- [Transformers](https://github.com/huggingface/transformers) — model loading, training, and inference
- [PEFT](https://github.com/huggingface/peft) — parameter-efficient fine-tuning (LoRA)
- [Datasets](https://github.com/huggingface/datasets) — data loading and preprocessing
- [SacreBLEU](https://github.com/mjpost/sacrebleu) — BLEU and chrF++ evaluation metrics
- [SentencePiece](https://github.com/google/sentencepiece) — subword tokenizer training
- [PyTorch](https://pytorch.org/) — deep learning framework

## 📄 License

The script `hy-mt2/train_dense.py` is based on code from [Tencent HunYuan](https://github.com/Tencent-Hunyuan/HunyuanLLM) and is distributed under the Apache 2.0 License.

## 📚 Citation

If you use this work, please cite:

```bibtex
@misc{khakas-mt,
  author       = {Adeshkin},
  title        = {Khakas–Russian Machine Translation},
  year         = {2025},
  url          = {https://github.com/adeshkin/khakas-mt}
}
```

# Task 1: News Topic Classifier Using BERT

**DevelopersHub Corporation — AI/ML Engineering Internship**

---

## Task Objective
Fine-tune a BERT transformer model to classify news headlines into 4 topic categories:
**World, Sports, Business, and Sci/Tech.**
Evaluate using accuracy and F1-score, then deploy the model using Gradio for live interaction.

---

## Dataset Used

| Property | Detail |
|---|---|
| Name | AG News |
| Source | Hugging Face Datasets (`load_dataset("ag_news")`) |
| Train samples | 120,000 (subset of 10,000 used for faster training) |
| Test samples | 7,600 (subset of 2,000 used for evaluation) |
| Classes | World (0), Sports (1), Business (2), Sci/Tech (3) |
| Balance | Perfectly balanced — 30,000 samples per class |

---

## Models Applied

| Component | Detail |
|---|---|
| Base model | `bert-base-uncased` |
| Architecture | BERT + Linear classification head (4 outputs) |
| Fine-tuning method | Hugging Face Trainer API |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Epochs | 3 |
| Batch size | 16 (train), 32 (eval) |
| Max token length | 128 |
| Mixed precision | fp16 (on GPU) |

---

## Key Results and Findings

### Performance Metrics
| Metric | Score |
|---|---|
| Accuracy | ~92–94% |
| F1 Score (weighted) | ~0.92–0.94 |

### Per-Class F1 Scores
| Category | F1 Score |
|---|---|
| World | ~0.91 |
| Sports | ~0.98 |
| Business | ~0.92 |
| Sci/Tech | ~0.91 |

### Key Findings
- **Transfer learning is highly effective** — fine-tuning pre-trained BERT for just 3 epochs achieves ~92-94% accuracy without training from scratch
- **Sports headlines are easiest to classify** — sport-specific vocabulary makes this category the most distinct (F1 ~0.98)
- **World and Sci/Tech are sometimes confused** — both categories can contain technical and global vocabulary, causing occasional misclassification
- **128 token limit is sufficient** — AG News headlines are short; no meaningful truncation occurs
- **Small subset performs well** — training on only 10,000 of 120,000 samples still achieves strong results, showing BERT's data efficiency
- **DataCollatorWithPadding** is more memory efficient than fixed padding for variable-length inputs

---

## Project Structure

```
task1/
│
├── Task1_News_Classifier_BERT.ipynb   # Main notebook — training & evaluation
├── app.py                              # Gradio deployment for live interaction
├── requirements.txt                    # All required Python packages
├── README.md                           # This file
│
├── saved_model/                        # Generated after running the notebook
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── vocab.txt
│
└── plots/                              # Generated during notebook execution
    ├── label_distribution.png
    ├── text_length_distribution.png
    ├── confusion_matrix.png
    └── f1_per_class.png
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd task1
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
Open `Task1_News_Classifier_BERT.ipynb` in Jupyter and run all cells top to bottom.
- Training takes ~5–10 min on GPU, ~30–40 min on CPU
- The model is automatically saved to `./saved_model/`

### 4. Launch Gradio app
```bash
python app.py
```
This opens a browser interface where you can type any news headline and get a live prediction.

---

## Requirements
- Python 3.8+
- See `requirements.txt` for all packages
- GPU recommended (but CPU works too, just slower)

---

## Disclaimer
Results may vary slightly depending on hardware, random seed, and whether the full dataset or subset is used

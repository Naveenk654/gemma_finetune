# Gemma-2-2B Hindi Instruct 🇮🇳

Fine-tuning Google Gemma-2-2B-IT for Hindi and Hinglish instruction following using QLoRA.

**Status:** Week 1 complete (June 2026) — dataset selection and methodology locked.

## Project goal
Build a small, open-source language model that responds fluently in Hindi/Hinglish to Hindi/Hinglish instructions. The base Gemma model understands Hindi but defaults to English responses; this project closes that gap.

## Approach summary
- **Base model:** `google/gemma-2-2b-it` (chosen via empirical tokenizer comparison)
- **Method:** QLoRA (4-bit quantization + LoRA adapters)
- **Compute:** Kaggle T4 x2 (free tier)
- **Evaluation:** LLM-as-judge on held-out prompts across 6 task categories

## Progress

### ✅ Week 1: Foundations + Dataset Selection
- Empirical tokenizer comparison across Qwen, Gemma, and Llama for Hindi
- Manually scored 75 random samples across 3 candidate Hindi instruction datasets
- Identified 11 distinct failure modes; designed filtering strategy for each
- Locked final training mix: ai4bharat wikihow + samvaad-hi-v1 + filtered alpaca-hindi

### 🔜 Week 2: Data Cleaning Pipeline (in progress)

### Upcoming
- Week 3: Evaluation harness + baseline
- Week 4-5: First training runs + iteration
- Week 6: Ablation experiments
- Week 7: Final model + Gradio demo
- Week 8: Documentation + blog post

## Why Gemma (not Llama or Qwen)
Empirical tokenizer efficiency comparison on Hindi text:

| Hindi sentence | Qwen tokens | Gemma tokens | Llama tokens |
|---------------|-------------|--------------|--------------|
| भारत एक विशाल देश है... | 54 | **17** | 27 |
| मुझे आज का मौसम बताओ | 19 | **8** | 11 |
| किसान को फसल बोने से पहले... | 45 | **20** | 28 |

Gemma uses 2-3× fewer tokens for Hindi by using meaningful Devanagari subwords; Qwen and Llama fall back to UTF-8 byte fragments. Better tokenization → more efficient learning during fine-tuning.

## Repo structure (in progress)
\`\`\`
.
├── notebooks/
│   └── 01_dataset_evaluation.ipynb   # Week 1: Tokenizer comparison + dataset quality scoring
├── data/                              # Cleaned data (Week 2)
├── src/                               # Training scripts (Week 4+)
└── README.md
\`\`\`

## License
Apache-2.0 (matching base model)

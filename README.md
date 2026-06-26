# Hindi/Hinglish Instruction Fine-tuning of Gemma-2-2B (QLoRA)

Fine-tuning [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) to respond **consistently in Hindi/Hinglish**, with a rigorous before/after evaluation.

🔗 **[Live Demo](https://huggingface.co/spaces/naveenk879/gemma-hindi-demo)** · **[Fine-tuned Model](https://huggingface.co/naveenk879/gemma-2-2b-hindi-merged)** · **[LoRA Adapter](https://huggingface.co/naveenk879/gemma-2-2b-hindi-lora)**

---

## Problem

Open small models like Gemma-2-2B can read Hindi, but respond **inconsistently** — they frequently default to English, especially when prompted in **Hinglish** (romanized Hindi). For an Indian-market assistant, this is a real failure: a user writing `"yaar mujhe ek movie suggest karo"` expects a Hindi reply, not English.

**Goal:** Use parameter-efficient fine-tuning (QLoRA) to make the model reply in Hindi/Hinglish reliably — and *measure* the improvement properly, rather than just eyeballing a few outputs.

> **Scope note (honest framing):** This project targets **language consistency**, not factual quality. As a 2B model, it has limited world knowledge and will hallucinate facts — an expected limitation of model size, separate from the language-consistency objective.

---

## Results

Evaluated on a **held-out set of 320 prompts** (100 each from three source datasets by category + 20 hand-written Hinglish prompts), comparing the base model vs. the fine-tuned model on the same prompts.

### Hindi-script consistency (fraction of response in Devanagari)

| Category      | Base  | Fine-tuned | Change   |
|---------------|-------|------------|----------|
| wikihow-hi    | 0.500 | 0.716      | **+0.216** |
| samvaad-hi    | 0.442 | 0.741      | **+0.299** |
| alpaca-hindi  | 0.290 | 0.753      | **+0.463** |
| **hinglish**  | **0.000** | **0.752** | **+0.752** |
| **Overall**   | **0.385** | **0.738** | **+0.353** |

### Response-level summary

| Metric                              | Base | Fine-tuned |
|-------------------------------------|------|------------|
| Responses mostly Hindi (>50%)       | 52%  | **97%**    |
| Responses mostly English (<30%)     | 46%  | **1%**     |

**Headline:** The base model replied to Hinglish prompts in **100% English** (0.000 Hindi). After fine-tuning, it replies **~75% in Hindi** — a capability that did not exist in the base model.

### Results data
Raw model responses for the full 320-prompt eval set are in `results/`:
- `results/base_model_response.json` — base Gemma-2-2B responses (baseline)
- `results/FineTune_model_responses.json` — fine-tuned model responses

The tables above are computed by comparing these two files on the same prompts.

### Honest limitations

- **Repetition increased** in 3 of 4 categories (a known side effect of fine-tuning small models on style; mitigated at inference with a repetition penalty).
- **Factual quality is limited** by the 2B base model (hallucinations) — not the target of this work.
- **LLM-as-judge** was built (Gemini, with position-bias mitigation) but full-scale judging was constrained by free-tier API rate limits; a sample directionally favored the fine-tuned model. The automatic metrics above are the primary quantitative evidence.

---

## Approach

### 1. Model selection — `selection-of-model-by-no.of_tokens.ipynb`
Compared tokenizer efficiency across candidate open models (Gemma vs. Qwen vs. Llama) on Hindi text. **Gemma-2-2B tokenized Hindi 2–3× more efficiently** (fewer tokens per sentence), making it the most cost-effective base for Hindi fine-tuning — the basis for choosing it.

### 2. Data cleaning & preparation — `data-cleaning-dataset1.ipynb`, `dataset2-3.ipynb`
- Combined **3 datasets**: `alpaca-cleaned-hindi`, `samvaad-hi-v1`, `ai4bharat/wikihow-hi`.
- Built **unit-tested cleaning filters**: Devanagari-ratio threshold, length bounds, identity/refusal removal, repetition-loop detection, harmful-content filtering (including script-mixed variants).
- **Per-dataset threshold tuning** — inspected each dataset's distribution independently rather than applying uniform cutoffs.
- Extracted **first-turn-only** from the multi-turn `samvaad` data to fit a single-turn instruction format.
- Final balanced mix: **40% wikihow / 40% samvaad / 20% alpaca** → **11,677 examples**, split 90/5/5 (train/val/test) with verified zero train/test overlap.

### 3. Evaluation — `dataset-evaluation-alpaca-hindi.ipynb`
- **320-prompt** held-out eval set (300 by-source + 20 hand-written Hinglish).
- Automatic metrics: **Hindi-script ratio**, response length, repetition score.
- Generated and saved **base-model responses first** to establish a true baseline before any training.
- **LLM-as-judge** pipeline (Gemini) with randomized A/B ordering to mitigate position bias.

### 4. Training (QLoRA)— `training.ipynb`
- **QLoRA**: 4-bit NF4 quantization + LoRA adapters (**r=16, α=32**, dropout 0.05) on q/k/v/o + gate/up/down projections.
- **20.7M trainable parameters (0.8%** of the model).
- bf16 compute, `paged_adamw_8bit`, cosine schedule, lr 2e-4, batch 4 × grad-accum 4, max-len 768.
- **Validation-loss early stopping**: monitored eval loss every 200 steps; it bottomed at **step 1200 (eval_loss 1.516)** then rose as training loss kept falling (classic overfitting). Selected the step-1200 checkpoint rather than the final one.

---

## Engineering notes

A few non-obvious issues solved during the project:

- **Gemma2 generation bug:** gradient checkpointing forces `use_cache=False`, which breaks Gemma2's attention during generation and produces degenerate, repeating-token output. Fix: before inference, set `model.config.use_cache=True`, `model.gradient_checkpointing_disable()`, `model.eval()`.
- **Hardware precision:** Gemma's internal activations can exceed float16's max (65504); float16-only GPUs (e.g. T4) produced corrupted output. Training on bf16-capable hardware (A40) resolved it.
- **Artifact discipline:** backed up the best checkpoint before `save_total_limit` rotation could delete it.

---

## Tech stack

Python · PyTorch · Hugging Face Transformers / PEFT / TRL · bitsandbytes · QLoRA · Gradio

## Model

- **Adapter:** [`naveenk879/gemma-2-2b-hindi-lora`](https://huggingface.co/naveenk879/gemma-2-2b-hindi-lora)
- **Merged model:** [`naveenk879/gemma-2-2b-hindi-merged`](https://huggingface.co/naveenk879/gemma-2-2b-hindi-merged)
- **Base:** [`google/gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it)

## License

The fine-tuned weights inherit the [Gemma license](https://ai.google.dev/gemma/terms). Code in this repository is released under the MIT License.

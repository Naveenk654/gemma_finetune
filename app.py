import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "naveenk879/gemma-2-2b-hindi-merged"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, device_map="cpu")
model.config.use_cache = True
model.eval()
print("Model loaded.")

def respond(message, history):
    messages = [{"role": "user", "content": message}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            do_sample=True, temperature=0.7, top_p=0.9,
            repetition_penalty=1.3, use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

demo = gr.ChatInterface(
    fn=respond,
    title="🪔 Gemma-2-2B Hindi/Hinglish Assistant",
    description=(
        "**A Gemma-2-2B model fine-tuned with QLoRA to respond consistently in Hindi/Hinglish.** "
        "Try writing in Hindi or Hinglish (e.g. 'mujhe chai banane ki vidhi batao').\n\n"
        "⚠️ This demonstrates *language-consistency* fine-tuning (Hindi usage 38%→74% overall, 0%→75% on Hinglish). "
        "As a small 2B model, it may produce factually inaccurate content (hallucinations) — an expected limitation of model size.\n\n"
        "ℹ️ Responses are capped at 256 tokens, so longer answers may end mid-sentence.\n\n"
        "⏳ Free CPU — responses take ~1-2 minutes."
    ),
    examples=[
        "मुझे चाय बनाने की विधि बताओ",
        "yaar mujhe weekend ke liye ek movie suggest karo",
        "exam ke liye kaise padhu? tips do",
        "subah jaldi uthne ke liye kya karu?",
    ],
)

if __name__ == "__main__":
    demo.launch()

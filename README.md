# 📰 BART News Summarizer

A fine-tuned **BART-base** model for **abstractive news summarization**.

Trained on the **CNN/DailyMail dataset** using Hugging Face **Transformers** and **PyTorch** in Google Colab, the model generates **concise and coherent summaries** of lengthy news articles — ideal for quick news consumption.

---

## 🚀 Key Highlights

| Metric | Value |
| :--- | :--- |
| **Model** | `facebook/bart-base` (fine-tuned) |
| **Dataset** | CNN/DailyMail (subset) |
| **ROUGE-1** | 24.6 |
| **ROUGE-2** | 9.5 |
| **ROUGE-Lsum** | 22.48 |
| **Best Checkpoint** | Epoch 2 |
| **Environment** | Google Colab (T4 GPU) |

---

## 💡 Example Usage

Use the following Python code snippet to load the fine-tuned model and generate a summary:

```python
from transformers import BartTokenizer, BartForConditionalGeneration

# Load the fine-tuned model (replace "path_to_finetuned_model" with your actual path or Hugging Face ID)
model = BartForConditionalGeneration.from_pretrained("path_to_finetuned_model")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# Example Text
text = "The president met with global leaders to discuss climate change and international trade. The summit, which lasted two days, concluded with a joint statement emphasizing the need for immediate action to reduce carbon emissions and promote sustainable economic practices across all participating nations."

# Tokenize and Generate Summary
inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
summary_ids = model.generate(
    inputs["input_ids"], 
    max_length=150, 
    min_length=40, 
    length_penalty=2.0,
    num_beams=4 # Adding num_beams for better quality
)

# Decode and Print
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

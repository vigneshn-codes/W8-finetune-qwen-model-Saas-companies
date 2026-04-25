# Fine-Tuning Qwen3-0.6B on SaaS Companies Data

Fine-tune a small LLM (Qwen3-0.6B) on a custom SaaS/B2B companies dataset using QLoRA + Unsloth on Google Colab, export it to GGUF, run it locally with Ollama, and expose it via a FastAPI endpoint.

---

## Project Structure

```
.
├── saas_model_lora_2.ipynb         # Main fine-tuning notebook (run on Google Colab)
├── saas_model_lora-2/              # Fine-tuned model artifacts
│   ├── adapter_config.json         # LoRA adapter configuration
│   ├── adapter_model.safetensors   # Trained LoRA weights
│   ├── tokenizer.json              # Tokenizer
│   ├── tokenizer_config.json
│   ├── chat_template.jinja         # Chat template for Ollama
│   ├── Modelfile                   # Ollama Modelfile
│   ├── Qwen3-0.6B.Q8_0.gguf       # Quantized model for local inference
│   └── companies_dataset.json      # Training dataset
├── saas-model-model-test/
│   └── app.py                      # FastAPI server wrapping the Ollama model
└── README_Fine-tune.md             # Theory reference (LLMs, LoRA, RAG vs Fine-tuning)
```

---

## Prerequisites

- [Google Account](https://accounts.google.com) — for Google Colab (free T4 GPU)
- [Ollama](https://ollama.com) — for running the model locally
- Python 3.10+
- `pip install fastapi uvicorn requests`

---

## Step-by-Step Guide

### Step 1 — Prepare Your Dataset

The training data is a JSON file of company records. Each record follows this structure:

```json
[
  {
    "Name": "GitHub",
    "URL": "https://github.com",
    "Type": "b2b, b2c, saas",
    "Category": "software development",
    "Keywords": "version control, collaboration, developer tools",
    "Description": "GitHub is a leading AI-powered developer platform..."
  }
]
```

Place your dataset at `saas_model_lora-2/companies_dataset.json` (or upload it directly to Colab in Step 4).

---

### Step 2 — Open Google Colab and Select a GPU

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Open `saas_model_lora_2.ipynb` (upload it or open from Google Drive)
3. Go to **Runtime → Change runtime type → T4 GPU** → Save

---

### Step 3 — Install Dependencies

Run the first notebook cell:

```python
!pip install unsloth
!pip install --no-deps xformers trl peft accelerate bitsandbytes
```

This installs:
- **Unsloth** — 2× faster training, 70% less VRAM
- **TRL** — Supervised Fine-Tuning (SFT) trainer
- **PEFT** — LoRA adapter support
- **BitsAndBytes** — 4-bit quantization

---

### Step 4 — Load the Base Model (QLoRA 4-bit)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-0.6B-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,        # auto-detects float16/bfloat16
    load_in_4bit = True  # QLoRA: loads base model in 4-bit
)
```

This downloads **Qwen3-0.6B** quantized to 4-bit (~600MB). The base model weights stay frozen — only the LoRA adapters will be trained.

---

### Step 5 — Attach LoRA Adapters

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                                              # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)
```

**Key parameters:**
| Parameter | Value | Meaning |
|-----------|-------|---------|
| `r` | 16 | Rank — controls adapter capacity (higher = more capacity, more VRAM) |
| `lora_alpha` | 16 | Scaling factor (keep equal to `r` as a safe default) |
| `target_modules` | q/k/v/o projections | Which attention layers to adapt |

---

### Step 6 — Format and Load the Dataset

```python
import json
from datasets import Dataset

with open('companies_dataset.json', 'r') as f:
    raw_data = json.load(f)

prompt_template = """Instruction: Provide information about the company.
Company Name: {}
Response: {}"""

def format_data(example):
    response_text = f"URL: {example['URL']}, Category: {example['Category']}, Description: {example['Description']}"
    return {"text": prompt_template.format(example['Name'], response_text)}

dataset = Dataset.from_list(raw_data).map(format_data)
```

Each training example teaches the model: given a company name, respond with its URL, category, and description.

---

### Step 7 — Train the Model

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        max_steps = 500,
        learning_rate = 2e-4,
        fp16 = True,
        logging_steps = 1,
        output_dir = "outputs",
    ),
)

trainer.train()
```

Training runs for **500 steps** (~10–20 minutes on Colab T4). Watch the loss — it should decrease steadily.

---

### Step 8 — Save the Model and Export to GGUF

```python
# Save LoRA adapter weights
model.save_pretrained("saas_model_lora")
tokenizer.save_pretrained("saas_model_lora")

# Export to GGUF (Q8_0 quantization — high quality, ~600MB)
model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method="q8_0")
```

Then download the output files to your machine:

```python
from google.colab import files
files.download("saas_model_lora.zip")
files.download("model_gguf_gguf/Qwen3-0.6B.Q8_0.gguf")
files.download("model_gguf_gguf/Modelfile")
```

---

### Step 9 — Load the Model into Ollama

1. Install Ollama from [ollama.com](https://ollama.com) if you haven't already.

2. Place the downloaded files in the `saas_model_lora-2/` directory:
   - `Qwen3-0.6B.Q8_0.gguf`
   - `Modelfile`

3. Create the Ollama model:

```bash
cd saas_model_lora-2
ollama create saas-orgs -f Modelfile
```

4. Test it in the terminal:

```bash
ollama run saas-orgs
```

Try a prompt like:
```
Company Name: Salesforce
```

---

### Step 10 — Run the FastAPI Server

The `saas-model-model-test/app.py` wraps the Ollama model in a REST API.

Start the server:

```bash
cd saas-model-model-test
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`.

**Endpoint:**

```
POST /advice
Content-Type: application/json

{
  "input": "Tell me about Salesforce"
}
```

**Example with curl:**

```bash
curl -X POST http://localhost:8000/advice \
  -H "Content-Type: application/json" \
  -d '{"input": "Tell me about Salesforce"}'
```

**Example response:**

```json
{
  "data": "URL: https://salesforce.com, Category: CRM, Description: Salesforce is a cloud-based CRM platform..."
}
```

You can also test it via the auto-generated docs at `http://localhost:8000/docs`.

---

## How It All Fits Together

```
companies_dataset.json
        │
        ▼
  Google Colab (Unsloth + QLoRA)
        │  fine-tune Qwen3-0.6B
        ▼
  LoRA Adapter weights (.safetensors)
        │
        │  export to GGUF
        ▼
  Qwen3-0.6B.Q8_0.gguf + Modelfile
        │
        │  ollama create
        ▼
  Ollama (local inference)
        │
        │  HTTP API
        ▼
  FastAPI → POST /advice
```

---

## Model Configuration Reference

| Setting | Value |
|---------|-------|
| Base model | `unsloth/Qwen3-0.6B-bnb-4bit` |
| LoRA rank (`r`) | 16 |
| LoRA alpha | 16 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Quantization (GGUF) | Q8_0 |
| Max sequence length | 2048 |
| Training steps | 500 |
| Learning rate | 2e-4 |
| Batch size | 2 (effective: 8 with grad accum) |
| Ollama model name | `saas-orgs:latest` |

---

## Further Reading

See `README_Fine-tune.md` for deep-dive theory on:
- LLM internals and Transformer architecture
- QKV attention explained
- LoRA vs QLoRA math and memory savings
- Full Fine-Tuning vs PEFT vs SFT comparison
- RAG vs Fine-Tuning decision framework

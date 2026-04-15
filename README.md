# # Week 8 -> Day 1 -> LLM, Transformers & Fine-Tuning 
---

## Table of Contents

**Foundations**
- [1. What is an LLM?](#1-what-is-an-llm)
- [2. Transformer Architecture](#2-transformer-architecture)
- [3. QKV — Query, Key, Value](#3-qkv--query-key-value)
- [4. Parameters vs Hyperparameters](#4-parameters-vs-hyperparameters)

**Fine-Tuning**
- [5. Fine-Tuning — What & Why?](#5-fine-tuning--what--why)
- [6. Fine-Tuning Types](#6-fine-tuning-types)
- [7. LoRA & QLoRA Deep Dive](#7-lora--qlora-deep-dive)

**Tools & Practice**
- [8. Unsloth — What & Why?](#8-unsloth--what--why)
- [9. Complete Colab Walkthrough](#9-complete-colab-walkthrough--fine-tuning-with-unsloth)
- [10. JSON Dataset Format](#10-json-dataset-format)

**RAG vs Fine-Tuning**
- [11. RAG vs Fine-Tuning](#11-rag-vs-fine-tuning)
- [12. Example Use Cases](#12-example-use-cases)

---

## 1. What is an LLM?

An **LLM (Large Language Model)** is a type of AI that has read billions of pages of text and learned the patterns of human language. Think of it as an incredibly well-read assistant that can write, summarize, translate, code, and reason — all by predicting what word comes next.

> **🍳 The Chef Analogy**
>
> Imagine a chef who has read *every cookbook ever written*. They haven't memorized every recipe word-for-word, but they deeply understand how flavors, ingredients, and techniques combine. When you ask them to create a dish, they draw on all that absorbed knowledge. An LLM works the same way with language.

### How does an LLM generate text?

An LLM works by predicting **one token at a time**. A token is roughly a word or part of a word. Given the beginning of a sentence, it calculates the probability of every possible next token and picks the most likely one. Then it uses that token plus the original context to pick the next one, and so on.

```
Input tokens (context)                              Predicted
┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐    ┌─────────┐
│  The  │→ │  cat  │→ │  sat  │→ │  on   │→ │  the  │ ─→ │ mat ✨  │
└───────┘  └───────┘  └───────┘  └───────┘  └───────┘    └─────────┘
```

### Popular LLMs

| Model | Company | Notable For |
|-------|---------|-------------|
| GPT-4o | OpenAI | Multimodal, widely used |
| Claude | Anthropic | Safety-focused, long context |
| Llama 4 | Meta | Open-source, highly capable |
| Gemini | Google | Integrated with Google services |
| Qwen 3 | Alibaba | Strong multilingual support |

---

## 2. Transformer Architecture

The **Transformer** is the engine inside every modern LLM. Introduced in the landmark 2017 paper "Attention Is All You Need," it replaced older sequence models (RNNs, LSTMs) with a mechanism called **self-attention** that lets the model look at all words simultaneously rather than one by one.

> **📚 The Library Analogy**
>
> Older models (RNNs) read a book page by page, trying to remember everything. A Transformer is like having the *entire book open on a giant table* — it can look at any page at any time and understand how chapter 1 relates to chapter 20 instantly.

### Transformer Building Blocks

A Transformer is made of stacked layers. Each layer has two main sub-layers: a **Multi-Head Self-Attention** block and a **Feed-Forward Neural Network** block. In between these are normalization and residual connections.

```
                ╔══════════════════════════════════════╗
                ║     ONE TRANSFORMER LAYER (×N)       ║
                ║                                      ║
                ║   ┌──────────────────────────────┐   ║
                ║   │    → Next Layer or Output    │   ║
                ║   └──────────────┬───────────────┘   ║
                ║                  │                   ║
                ║   ┌──────────────┴───────────────┐   ║
                ║   │       Add & Layer Norm       │   ║
                ║   └──────────────┬───────────────┘   ║
                ║                  │                   ║
                ║   ┌──────────────┴───────────────┐   ║
                ║   │    Feed-Forward Network      │   ║
                ║   └──────────────┬───────────────┘   ║
                ║                  │                   ║
                ║   ┌──────────────┴───────────────┐   ║
                ║   │       Add & Layer Norm       │   ║
                ║   └──────────────┬───────────────┘   ║
                ║                  │                   ║
                ║   ┌──────────────┴───────────────┐   ║
                ║   │ Multi-Head Self-Attention    │   ║
                ║   │        (QKV)                 │   ║
                ║   │   ┌─────┐ ┌─────┐ ┌─────┐    │   ║
                ║   │   │  Q  │ │  K  │ │  V  │    │   ║
                ║   │   └─────┘ └─────┘ └─────┘    │   ║
                ║   └──────────────┬───────────────┘   ║
                ║                  │                   ║
                ║   ┌──────────────┴───────────────┐   ║
                ║   │  Input Embedding + Position  │   ║
                ║   └──────────────────────────────┘   ║
                ║                                ×N    ║
                ║          (e.g., 32 layers            ║
                ║            in Llama 8B)              ║
                ╚══════════════════════════════════════╝
```

### Key Insight: Self-Attention

Self-attention lets each word look at every other word in the sentence and decide how much to "pay attention" to it. In the sentence "The *bank* of the *river* was steep," the word "bank" attends heavily to "river" to understand it means a riverbank, not a financial bank.

---

## 3. QKV — Query, Key, Value

The attention mechanism works through three vectors called **Q (Query)**, **K (Key)**, and **V (Value)**. These are the heart of how Transformers understand context.

> **🔍 The Google Search Analogy**
>
> Think of it like searching the internet. You type a **Query** (what you're looking for). Each web page has a title — that's the **Key**. Google compares your Query to all Keys and ranks them by relevance. Then you read the **Value** (the actual page content) of the best matches. That's exactly how QKV attention works!

### The Three Vectors

| Vector | Symbol | Role | Simple Question |
|--------|--------|------|-----------------|
| **Query** | Q | What am I looking for? | Each word generates a Query — the question that word asks about its context |
| **Key** | K | What do I contain? | Each word generates a Key — a descriptor of what information it carries |
| **Value** | V | Here's my actual info | The actual content that gets passed forward once a word is deemed relevant |

### The Attention Formula

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √dₖ) · V
```

**Step-by-step breakdown:**

| Step | Operation | What it does |
|------|-----------|-------------|
| Step 1 | `Q · Kᵀ` | Compute similarity between all word pairs |
| Step 2 | `÷ √dₖ` | Scale down to prevent extremely large values |
| Step 3 | `softmax` | Convert to probabilities (0 to 1, sum = 1) |
| Step 4 | `× V` | Weighted sum of actual values |

### Worked Example

**Sentence:** "The cat drank the milk"

When processing the word **"drank"**, the attention mechanism assigns scores to all other words:

```
Word:      The      cat     drank     the      milk
Score:    0.05     0.40     0.10     0.05     0.40
Bar:      ██       ████████ ███      ██       ████████
           low      HIGH     self     low      HIGH
```

"Cat" gets a high score (because cat is doing the drinking) and "milk" also gets a high score (because milk is what's being drunk). Words like "the" get low scores. This helps the model understand who did what to whom.

---

## 4. Parameters vs Hyperparameters

These two terms sound similar but are fundamentally different. Understanding the distinction is essential for anyone working with LLMs.

### Parameters — What the model LEARNS

These are the internal numbers (weights and biases) that the model adjusts during training. When we say "Llama 3 has 8 billion parameters," we mean it has 8 billion learned numerical values that encode its understanding of language.

**Analogy:** The knowledge stored in a student's brain after years of studying. You can't directly set this — it's learned from experience.

**Examples:** Attention weights in each layer, Feed-forward network weights, Embedding vectors for each token, Output projection weights

### Hyperparameters — What YOU SET before training

These are the settings and knobs you configure before training starts. They control *how* the model learns, not *what* it learns. Choosing good hyperparameters is part art, part science.

**Analogy:** The study plan — how many hours per day, which subjects to focus on, what textbooks to use. These are decisions made by the teacher, not the student.

**Examples:** Learning rate (e.g., 2e-4), Batch size (e.g., 4), Number of epochs (e.g., 3), LoRA rank (e.g., 16)

### Common Hyperparameters Explained

| Hyperparameter | What It Controls | Typical Value | Simple Analogy |
|---------------|-----------------|---------------|----------------|
| **Learning Rate** | How big each learning step is | 1e-4 to 5e-5 | Walking speed while exploring — too fast = you overshoot, too slow = takes forever |
| **Batch Size** | How many examples to learn from at once | 2 – 32 | Number of practice problems done before checking answers |
| **Epochs** | Number of full passes through the data | 1 – 3 | How many times you re-read the textbook |
| **Max Seq Length** | Maximum input length the model can handle | 2048 – 8192 | How long a passage the student can read at once |
| **LoRA Rank (r)** | Capacity of the adapter layers | 8 – 64 | How many notebook pages available for new notes |
| **LoRA Alpha** | Scaling factor for LoRA updates | 16 – 32 | How boldly the new notes are written |
| **Warmup Steps** | Gradual increase of learning rate at start | 5 – 100 | Warming up before sprinting |

---

## 5. Fine-Tuning — What & Why?

**Fine-tuning** is the process of taking a pre-trained LLM (one that already understands language) and training it further on your specific data to customize its behavior for your particular use case.

> **🎓 The University Graduate Analogy**
>
> A pre-trained model is like a university graduate with broad knowledge. Fine-tuning is like that graduate doing a **specialized residency** — they already understand medicine in general, but now they're becoming an expert cardiologist by training on heart-specific cases.

### Why Fine-Tune?

- 🧠 **Inject Domain Knowledge** — Medical, legal, financial
- 🎭 **Custom Personality** — Tone, style, branding
- 🎯 **Task Optimization** — Classification, extraction
- 💰 **Cost Reduction** — Smaller model, same quality

### Real-World Fine-Tuning Examples

#### 🏥 Healthcare — Clinical Note Summarization

**Problem:** Doctors spend hours writing summaries of patient encounters. Generic LLMs produce summaries that miss clinical nuances.

**Solution:** Fine-tune Llama 8B on 50,000 de-identified clinical note → summary pairs. The model learns medical abbreviations (PRN, BID, q4h), proper SOAP note format, and which clinical details matter.

**Result:** 70% reduction in documentation time. Summaries are clinically accurate and follow hospital formatting guidelines.

#### ⚖️ Legal — Contract Analysis

**Problem:** Law firms need to review thousands of contracts to identify risky clauses, non-standard terms, and missing provisions.

**Solution:** Fine-tune on 20,000 annotated contract clauses with labels for risk levels and clause types. The model learns domain-specific language like indemnification, force majeure, and liability caps.

**Result:** 90% accuracy in identifying risky clauses, reducing review time from hours to minutes per contract.

#### 🛒 E-Commerce — Product Description Generator

**Problem:** Writing unique product descriptions for thousands of SKUs is expensive and slow.

**Solution:** Fine-tune on 100,000 existing high-converting product descriptions paired with product attributes (category, features, specs). The model learns the brand voice, SEO patterns, and how to highlight selling points.

**Result:** Generates on-brand descriptions in seconds, increasing SEO traffic by 35% and reducing content costs by 80%.

#### 💻 Code — Company-Specific Code Assistant

**Problem:** Generic coding assistants don't know your company's internal APIs, coding standards, or architectural patterns.

**Solution:** Fine-tune on the company's codebase, pull requests, and code review comments. The model learns internal API usage patterns, naming conventions, and preferred design patterns.

**Result:** Code suggestions that actually compile and follow team standards. 40% faster onboarding for new developers.

---

## 6. Fine-Tuning Types

### 6.1 Full Fine-Tuning (FFT)

**All** parameters in the model are updated during training. This is the most powerful but also the most resource-intensive method.

```
┌─────────────────────────────────────────────────────────┐
│         ALL LAYERS TRAINABLE — Every weight updated      │
│                    (100% of parameters)                  │
└─────────────────────────────────────────────────────────┘
  ✓ Maximum performance    ✗ Huge GPU memory    ✗ Risk of catastrophic forgetting
```

**When to use:** When you have massive compute (multiple A100 GPUs), a very large dataset, and need the model to deeply learn new behavior. Rarely needed — LoRA often achieves comparable results.

**Example:** Meta fine-tuning base Llama to create Llama-Chat. They had thousands of GPUs and millions of instruction-following examples.

### 6.2 Partial Fine-Tuning

Only **selected layers** of the model are updated (typically the last few layers or the classification head), while the rest remain frozen.

```
┌─────────────────────────────────────────┐
│    Layers 1-24: FROZEN 🔒              │  ░░░░░░░ (unchanged)
├─────────────────────────────────────────┤
│    Layers 25-28: FROZEN 🔒             │  ░░░░░░░ (unchanged)
├─────────────────────────────────────────┤
│    Layers 29-32: TRAINABLE 🔓          │  ████████ (updated)
├─────────────────────────────────────────┤
│    Output Head: TRAINABLE 🔓           │  ████████ (updated)
└─────────────────────────────────────────┘
```

**When to use:** When you want more control than PEFT but less compute than FFT. Common in computer vision (freeze backbone, train head) but less common in NLP today because LoRA is more flexible.

**Example:** Freezing the embedding and early transformer layers of a model and only fine-tuning the last 4 layers for a sentiment classification task.

### 6.3 Parameter-Efficient Fine-Tuning (PEFT) / LoRA

PEFT methods add a **small number of trainable parameters** while keeping the original model frozen. The most popular PEFT method is **LoRA** (Low-Rank Adaptation). Only ~1-5% of parameters are trained.

```
┌───────────────────────────────────────┐    ┌──────────────────┐
│  Original Model Weights — FROZEN 🔒   │ +  │  LoRA Adapters  │
│           (99% of params)             │    │  TRAINABLE 🔓   │
│                                       │    │  (1% of params)  │
└───────────────────────────────────────┘    └──────────────────┘

✓ 70-90% less VRAM   ✓ Fast training   ✓ Near-FFT quality   ✓ Easy to swap/merge
```

**When to use:** Almost always! LoRA is the default go-to for fine-tuning. It's efficient, effective, and you can train on a single consumer GPU. Use QLoRA (4-bit quantized base + LoRA) for even less memory.

**Example:** Fine-tuning Llama 3.1 8B on a custom dataset using QLoRA on Google Colab's T4 GPU with only 15GB VRAM.

### 6.4 Instruction Fine-Tuning / Supervised Fine-Tuning (SFT)

This is not about *which parameters* to train, but about *what data format* to use. SFT trains the model on input-output pairs: given an instruction, produce the expected response.

**Data Format for SFT:**

```json
{
  "instruction": "Summarize the following patient note...",
  "input": "Patient presents with acute chest pain...",
  "output": "65yo male with acute MI, started on..."
}
```

SFT is how base models become **chat models**. Base Llama 3 just predicts next tokens. After SFT on instruction-following data, it becomes Llama 3 Instruct — understanding prompts and giving helpful responses.

**When to use:** When you want your model to follow instructions, answer questions, or perform specific tasks in a consistent format. This is the most common type of fine-tuning for chatbots and task-specific models.

**Example:** Training a customer support bot on 10,000 examples of customer questions paired with ideal agent responses.

### Comparison Summary

| Type | Params Trained | VRAM Needed | Best For |
|------|---------------|-------------|----------|
| **Full Fine-Tuning** | 100% | Very High (4-8× A100s) | When nothing else works & you have massive compute |
| **Partial Fine-Tuning** | 10-30% | High | Classification tasks, feature extraction |
| **PEFT / LoRA** | 1-5% | Low (single GPU) | Almost everything — the default choice |
| **SFT (Instruction)** | Varies (usually with LoRA) | Depends on method | Teaching models to follow instructions |

---

## 7. LoRA & QLoRA Deep Dive

### LoRA — Low-Rank Adaptation

Instead of updating the entire weight matrix **W** (which might be 4096 × 4096 = 16 million numbers), LoRA decomposes the update into two small matrices: **A** (4096 × r) and **B** (r × 4096), where **r** (the rank) is typically 8-64. Only A and B are trained.

```
┌─────────────────┐         ┌─────┐   ┌─────────────────┐         ┌─────────────────┐
│                 │         │     │   │                 │         │                 │
│       W         │    +    │  A  │ × │        B        │    =    │    W + ΔW       │
│  (4096×4096)    │         │4096 │   │    (r×4096)     │         │  Adapted Model  │
│   FROZEN 🔒     │         │ ×r  │   │  TRAINABLE 🔓   │         │                 │
│                 │         │     │   │                 │         │                 │
└─────────────────┘         └─────┘   └─────────────────┘         └─────────────────┘
   Original Weights          LoRA Adapters (TRAINABLE)               Final Output

If r = 16: trainable params = 4096×16 + 16×4096 = 131K (vs 16.7M full) → 99.2% reduction!
```

### QLoRA — Quantized LoRA

QLoRA takes LoRA one step further. It **quantizes the frozen base model to 4-bit** precision, dramatically cutting memory usage while keeping the LoRA adapters in 16-bit for accurate learning.

### Memory Savings Comparison

| Method | Base Model Precision | VRAM for Llama 8B | Can Run On |
|--------|---------------------|-------------------|------------|
| Full Fine-Tuning (16-bit) | 16-bit | ~60 GB | 4× A100 |
| LoRA (16-bit) | 16-bit | ~18 GB | A100 / RTX 4090 |
| **QLoRA (4-bit)** | **4-bit** | **~6 GB** | **Colab T4!** |

---

## 8. Unsloth — What & Why?

**Unsloth** is an open-source library that makes fine-tuning LLMs **2× faster** and uses **70% less VRAM** — with zero accuracy loss. It achieves this through custom CUDA kernels and intelligent memory optimization.

> **🦥 Why Unsloth?**
>
> Without Unsloth, fine-tuning Llama 3.1 8B with QLoRA needs ~16GB VRAM and takes hours. With Unsloth, the same task needs ~6GB and finishes 2× faster. This means you can fine-tune powerful models on **Google Colab's T4 GPU** (15GB VRAM).

### Key Features

- ⚡ 2× Faster Training
- 📉 70% Less VRAM
- 🧠 500+ Models Supported
- 🔧 LoRA / QLoRA / FFT
- 📊 Vision, TTS, Embedding support
- 🆓 Colab/Kaggle
- 📦 Export to GGUF/Ollama
- 🎯 No Accuracy Loss

### How Unsloth Achieves Speed

Unsloth writes custom Triton/CUDA kernels for key operations (attention, cross-entropy loss, RoPE embeddings) that are specifically optimized for training. It also uses intelligent memory management to avoid redundant copies of data on the GPU.

---

## 9. Complete Colab Walkthrough — Fine-Tuning with Unsloth

> **📋 Prerequisites:** A Google account (for Colab), a Hugging Face account (to download models and optionally upload your fine-tuned model). That's it!

### Step 1: Open Google Colab & Select GPU

Go to **colab.research.google.com**. Create a new notebook. Then go to **Runtime → Change runtime type → T4 GPU** . Click Save.

### Step 2: Install Unsloth

```python
# Install Unsloth (takes ~2-3 minutes)
%%capture
!pip install unsloth
# Also install vllm if you want faster inference
!pip install vllm
```

### Step 3: Load the Base Model with QLoRA (4-bit)

We'll use Llama 3.1 8B Instruct as our base model. Unsloth automatically handles the 4-bit quantization.

```python
from unsloth import FastLanguageModel
import torch

# Configuration
max_seq_length = 2048
dtype = None          # Auto-detect (float16 / bfloat16)
load_in_4bit = True   # QLoRA: 4-bit quantization

# Load model + tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```

### Step 4: Add LoRA Adapters

Configure which layers to apply LoRA to and set the rank (r). Higher rank = more capacity but more memory.

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA rank (8-64 typical, 16 is a good default)
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 16,        # Scaling factor (usually = r)
    lora_dropout = 0,        # 0 is optimized by Unsloth
    bias = "none",
    use_gradient_checkpointing = "unsloth",  # Saves 30% more VRAM
    random_state = 3407,
)
```

### Step 5: Prepare Your JSON Dataset

Create your training data in a JSON format. For instruction fine-tuning, each example needs an instruction and a response. Here we use the Alpaca format:

```python
from datasets import load_dataset

# Option A: Load a Hugging Face dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# Option B: Load YOUR OWN local JSON dataset
# dataset = load_dataset("json", data_files="my_data.json", split="train")

# Format into the chat template
alpaca_prompt = """Below is an instruction. Write a response.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for inst, inp, out in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(inst, inp, out) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
```

### Step 6: Configure Training & Train!

```python
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,            # Use num_train_epochs=1 for full run
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        output_dir = "outputs",
    ),
)

# 🚀 Start training!
trainer_stats = trainer.train()
```

### Step 7: Test Your Fine-Tuned Model

```python
# Switch to inference mode
FastLanguageModel.for_inference(model)

inputs = tokenizer([
    alpaca_prompt.format(
        "Explain quantum computing in one paragraph.",
        "",   # no additional input
        "",   # leave response blank for generation
    )
], return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.batch_decode(outputs))
```

### Step 8: Save & Export the Model

```python
# Save LoRA adapter only (small, ~50MB)
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# Save merged 16-bit model
model.save_pretrained_merged("model_merged", tokenizer, save_method="merged_16bit")

# Export to GGUF for Ollama (quantized for local inference)
model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method="q4_k_m")

# Push to Hugging Face Hub
model.push_to_hub_merged("your-username/my-model", tokenizer, token="hf_...")
```

---

## 10. JSON Dataset Format

Your training data needs to be structured so the model can learn from input-output pairs. Here are the common formats:

### Alpaca Format (Single-turn)

Best for simple instruction → response tasks. Each example has an instruction, optional input context, and expected output.

```json
[
  {
    "instruction": "Classify the sentiment of this review",
    "input": "This product is amazing! Best purchase I've ever made.",
    "output": "Positive"
  },
  {
    "instruction": "Translate the following to French",
    "input": "The weather is beautiful today.",
    "output": "Le temps est magnifique aujourd'hui."
  },
  {
    "instruction": "What is the capital of Japan?",
    "input": "",
    "output": "The capital of Japan is Tokyo."
  }
]
```

### ShareGPT / ChatML Format (Multi-turn)

Best for conversational fine-tuning. Supports multi-turn conversations with system, user, and assistant roles.

```json
[
  {
    "conversations": [
      {"from": "system", "value": "You are a helpful medical assistant."},
      {"from": "human", "value": "What are symptoms of diabetes?"},
      {"from": "gpt", "value": "Common symptoms include increased thirst..."},
      {"from": "human", "value": "How is it diagnosed?"},
      {"from": "gpt", "value": "Diagnosis typically involves blood glucose tests..."}
    ]
  }
]
```

### Custom JSON

You can use any JSON structure — just map the fields in your formatting function:

```json
[
  {
    "question": "What is photosynthesis?",
    "answer": "Photosynthesis is the process by which plants convert...",
    "category": "biology"
  }
]
```

```python
# Then in your formatting function:
def format_fn(examples):
    texts = []
    for q, a in zip(examples["question"], examples["answer"]):
        text = f"Question: {q}\nAnswer: {a}{EOS_TOKEN}"
        texts.append(text)
    return {"text": texts}
```

> **💡 Dataset Quality Tips:** Aim for at least 500-1,000 examples for noticeable results. Quality matters far more than quantity — 1,000 excellent examples beat 100,000 mediocre ones. Always include diverse examples covering edge cases. Remove duplicates and inconsistencies.

---

## 11. RAG vs Fine-Tuning

**RAG (Retrieval-Augmented Generation)** and **Fine-Tuning** are two fundamentally different approaches to making LLMs smarter for your use case. They are complementary — not competing — strategies.

### RAG (Retrieval-Augmented Generation)

The model's weights stay *unchanged*. Instead, you give it relevant documents at query time. A retrieval system searches your knowledge base and injects relevant chunks into the prompt.

**How it works:**
1. User asks a question
2. System searches your documents
3. Top-K relevant chunks are retrieved
4. Chunks are injected into the prompt
5. LLM generates answer using the context

**Strengths:**
- ✓ No GPU training required
- ✓ Data can be updated instantly
- ✓ Traceable sources (citations)
- ✓ Works with any LLM out of the box

**Weaknesses:**
- ✗ Limited by context window size
- ✗ Can't change model behavior/tone
- ✗ Retrieval errors lead to bad answers
- ✗ Higher latency per query

### Fine-Tuning

The model's weights are *updated* to bake knowledge and behavior directly into the model. After training, the model inherently "knows" the new information and style.

**How it works:**
1. Prepare training dataset
2. Fine-tune model on your data
3. Model weights are permanently updated
4. Deploy the fine-tuned model
5. Model answers from internal knowledge

**Strengths:**
- ✓ Changes model behavior & personality
- ✓ Faster inference (no retrieval step)
- ✓ Learns patterns, not just facts
- ✓ Can learn new skills (e.g., coding style)

**Weaknesses:**
- ✗ Requires GPU and training time
- ✗ Updating knowledge = retraining
- ✗ Risk of hallucination on rare facts
- ✗ Can't easily cite sources

### Decision Framework: When to Use What?

| Scenario | Best Approach | Why |
|----------|--------------|-----|
| Company knowledge base Q&A with frequently changing docs | **RAG** | Documents change often; RAG handles dynamic data without retraining |
| Customer support bot with specific brand voice | **Fine-Tuning** | Needs consistent tone/personality baked into the model |
| Legal contract review with citeable references | **RAG** | Lawyers need source citations; RAG provides traceable references |
| Medical diagnosis assistant with specialized terminology | **Both** | Fine-tune for medical reasoning + RAG for latest guidelines |
| Code generation in company's internal framework | **Fine-Tuning** | Model needs to learn coding patterns and API usage, not just reference docs |
| Chatbot for a news aggregation platform | **RAG** | News changes hourly; impossible to retrain that frequently |
| Translating content into a company's specific style guide | **Fine-Tuning** | Style is a learned behavior, not retrievable knowledge |
| Enterprise search over 1M+ internal documents | **RAG** | Too much data to bake into model weights; retrieval scales better |

> **🏆 Pro Tip:** The best production systems often use **both together**. Fine-tune the model to understand your domain and speak in your style, then use RAG to ground it in the latest factual data. This gives you behavioral customization + up-to-date accuracy.

---

## 12. Example Use Cases

### 🏥 RAG Example: Hospital Knowledge Base

**Scenario:** A hospital has 50,000+ medical protocols, drug interaction databases, and treatment guidelines that update weekly.

**Why RAG:** The information changes too frequently to retrain. Doctors need citations back to the specific protocol document. The model doesn't need to "learn" medicine — it just needs to surface the right document at the right time.

**Architecture:** Embed all documents → Store in vector DB (Pinecone/Weaviate) → At query time, retrieve top-5 relevant chunks → Inject into prompt → LLM generates answer with source citations.

### 💬 Fine-Tuning Example: Brand-Voice Chatbot

**Scenario:** A D2C skincare brand wants a chatbot that sounds like their brand — warm, emoji-friendly, uses skincare terminology correctly, and never makes medical claims.

**Why Fine-Tuning:** Brand voice is a behavioral trait, not factual knowledge. You can't "retrieve" a personality. The model needs to fundamentally change how it communicates.

**Approach:** Collect 5,000 examples of ideal conversations between customers and the brand's best support agents → Format as ShareGPT conversations → Fine-tune Llama 3.1 8B using QLoRA on Unsloth → Deploy via vLLM.

### 🔬 Both Example: Pharmaceutical Research Assistant

**Scenario:** A pharma company needs an AI that can analyze clinical trial data, understand drug mechanisms, and answer researcher questions with references to specific papers.

**Why Both:** Fine-tune for deep understanding of pharmacology, chemical structures, and clinical trial methodology (behavioral knowledge). Use RAG for the latest published papers, FDA filings, and trial results (factual, changing data).

**Architecture:** Fine-tuned domain model + RAG pipeline over PubMed + internal trial databases. The fine-tuned model better understands the retrieved content and produces more accurate analysis.

---

## 🎯 Quick Reference

| Need | Best Approach |
|------|--------------|
| Change **WHAT** the model knows | → Fine-Tuning |
| Change **HOW** the model behaves | → Fine-Tuning |
| Dynamic, up-to-date facts | → RAG |
| Citeable source references | → RAG |
| All of the above | → Both! |

---


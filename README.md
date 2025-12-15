# PrimeGPT — Training a GPT-style Language Model from Scratch

I built **PrimeGPT** as a from-scratch implementation and training of a GPT-style autoregressive language model. My goal was to deeply understand how modern Large Language Models work internally.

I focused on **architecture**, **training dynamics**, **tokenization**, **generation behavior**, and **instruction-conditioning infrastructure**, rather than maximizing benchmark performance.

My core objective was to **build and train an LLM from scratch**, understand its failure modes, and reason about why instruction-following requires aligned data.

---

## 1. Project Goals

My goals for this project were:

-   To implement a GPT-style Transformer **from scratch**.
-   To train a **custom tokenizer**.
-   To build a **scalable training pipeline**.
-   To train on a large real-world text corpus.
-   To implement modern LLM training techniques.
-   To design an **instruction-tuning–ready training pipeline**.
-   To understand **why instruction-following fails without paired data**.

This was an **engineering- and learning-driven project** for me, not a benchmark-oriented one.

---

## 2. Model Architecture

I decided that PrimeGPT should follow the standard **decoder-only Transformer** architecture used in GPT-family models.

### High-level structure

```mermaid
graph TD
    A[Input Tokens] --> B[Token Embedding + Positional Embedding]
    B --> C[N Transformer Blocks]
    C --> D[Final LayerNorm]
    D --> E[Linear LM Head (weight-tied)]
    E --> F[Next-token logits]
```

### Core components

-   **Decoder-only Transformer**
-   **Causal self-attention**
-   **Pre-normalization blocks**
-   **Residual connections**
-   **Weight tying between embedding and LM head**

### Configurable dimensions

I defined all architectural hyperparameters in `config.py`, including:

-   Vocabulary size (from tokenizer)
-   Model dimension (`d_model`)
-   Number of Transformer layers
-   Number of attention heads
-   Maximum sequence length
-   Feedforward expansion ratio

This allowed me to easily scale the model without architectural changes.

---

## 3. Tokenizer

I trained a **SentencePiece BPE tokenizer** from scratch on the lyrics corpus.

### Tokenizer properties

-   **Model type**: BPE
-   **Vocabulary size**: 8,000 tokens
-   **Character coverage**: 99.95%
-   **Special tokens explicitly defined**

### Custom tokens

To support instruction-style conditioning and structured generation, I included the following in the tokenizer:

#### Control tokens
` <prompt> <song> </song> `

#### Genre tokens
```text
<genre_pop>
<genre_rap>
<genre_rock>
<genre_rb>
```

#### Emotion tokens
```text
<happy>
<sad>
<angry>
<romantic>
```

#### Structural tokens
```text
[intro]
[verse]
[chorus]
[bridge]
[outro]
```

These tokens enabled me to implement:

-   Prompt conditioning
-   Section-aware generation
-   Control token masking
-   Instruction-style formatting

---

## 4. Dataset Construction

I constructed the training data from a large lyrics corpus.

### Preprocessing

-   I dropped non-ASCII samples (to reduce tokenizer noise).
-   I converted all lyrics to plain text.
-   I randomly assigned genre and emotion labels.
-   I wrapped samples in a structured instruction-like format.

### Instruction-style formatting (infrastructure)

I formatted each training sample as:

```text
<prompt> <genre> <emotion> instruction text
<song>
lyrics...
</song>
```

> **⚠️ Important**
> While this resembles instruction tuning, the dataset does not contain true instruction–response pairs.
> The prompt is synthetic and weakly correlated with the lyrics.

---

## 5. Training Objective

I trained PrimeGPT using standard next-token prediction with cross-entropy loss.

### Instruction-aware loss masking

I designed the training pipeline to include instruction-tuning–compatible loss masking:

-   Tokens before `<song>` are masked.
-   Prompt tokens do not contribute to loss.
-   Only song content is supervised.
-   Prompts act purely as conditioning context.

This design matches how instruction tuning is typically implemented in aligned datasets.

-   ✅ The pipeline fully supports instruction tuning.
-   ❌ The dataset does not provide aligned supervision.

---

## 6. Training Techniques Used

I intentionally incorporated modern LLM training techniques into this project.

### Mixed Precision Training (AMP)
-   Used `torch.autocast` + `GradScaler`.
-   Reduces memory usage.
-   Increases throughput on consumer GPUs.

### Gradient Accumulation
-   Simulates large batch sizes.
-   Prevents GPU OOM.
-   Improves optimization stability.

### Gradient Clipping
-   Prevents exploding gradients.
-   Stabilizes deep Transformer training.

### Learning Rate Scheduling
-   Linear warmup.
-   Cosine decay.
-   Matches GPT-family training dynamics.

### Checkpointing and Resume
I optimized the checkpointing system to save:
-   Model weights
-   Optimizer state
-   Scheduler state
-   AMP scaler

Training can resume exactly from any checkpoint.

---

## 7. Generation System

I developed a fully custom generation pipeline that includes:

### Sampling controls
-   Temperature
-   Top-p (nucleus sampling)
-   Optional top-k
-   Repetition penalty

### Control-token handling
I implemented hard banning of control tokens during generation:
-   `<prompt>`
-   `<song>`
-   genre/emotion tokens

This prevents token leakage into generated lyrics.

### Early stopping
-   Generation stops automatically on `</song>`.

### Temperature decay
-   Temperature decays after `[chorus]`.
-   This encourages coherent endings and reduced drift.

---

## 8. Prompt Adherence Evaluation

I implemented a quantitative prompt adherence score:

-   It measures word overlap between prompt and generated lyrics.
-   I used it purely for diagnostics and analysis.
-   It is not used to modify gradients.

This revealed a key insight for me:
**Low loss and fluent text do not imply instruction-following.**

---

## 9. Key Findings and Learnings

### 1. Instruction tuning is fundamentally a data problem

Despite having prompt conditioning, loss masking, prompt dropout, and generation constraints, I observed that the model frequently ignored the prompt. This is expected behavior for base LMs trained without aligned data.

### 2. The pipeline supports instruction tuning — the data does not

My training code fully supports instruction tuning (Prompt/context separation, Output-only supervision, Control token handling, etc.). However:
-   Prompts were synthetic.
-   Lyrics were not written in response to prompts.
-   There was no strong mutual information between prompt and output.

I realized that the optimal likelihood solution is to ignore the prompt.

### 3. Loss is not a proxy for alignment

The model achieved low loss and fluent, coherent lyrics, yet failed to follow prompts. This reinforces that:
-   Likelihood optimization ≠ intent following.
-   Alignment must be learned from data.

### 4. Instruction-following cannot be hacked in

I learned that no amount of masking, penalties, temperature tricks, or token banning can replace properly aligned (instruction, response) pairs.

---

## 10. Why the Project Ends Here

My primary objective was to build and understand an LLM from scratch. **I fully achieved that goal.**

True instruction tuning would require:
-   Paired instruction–response datasets.
-   Synthetic data generation.
-   Or RLHF-style pipelines.

At that point, the problem shifts from LLM engineering to dataset alignment, which is outside the scope of this project. **I made a deliberate and informed decision to stop here.**

---

## 11. Summary

With PrimeGPT, I demonstrated:
-   Transformer internals.
-   Tokenizer training.
-   Large-scale autoregressive training.
-   Modern optimization techniques.
-   Instruction-tuning–ready infrastructure.
-   Empirical analysis of prompt failure modes.

I prioritized understanding over superficial performance, and this project serves as a strong foundation for my future work in LLM research, Applied ML engineering, and Instruction tuning and alignment.

---

## 12. Final Note

I built this project to understand how LLMs work — **and why they fail.**

That understanding is the real result.
# Prime_LLM Architecture Notes

These are the notes I made to document my understanding of the Transformer architecture while building PrimeGPT.

The entire process roughly follows this flow:

## Step 1: Tokenizer
I use the tokenizer to convert raw text (words/sub-words) into integers. It essentially maps every unique word or sub-word to a unique integer ID from the vocabulary.

## Step 2: Token Embeddings
This step converts the integer array into continuous vectors. Each token is mapped to a unique $n$-dimensional vector so the model can learn semantic relationships. 

For example, my model might learn:
- `love`  = `[0.89, 0.22, -0.14, ...]`
- `adore` = `[0.91, 0.18, -0.11, ...]`

These two vectors would be close in Euclidean space (and have high cosine similarity). This proximity in vector space is how the LLM "knows" these words have similar meanings.

## Step 3: Positional Embeddings
Since the Transformer is permutation-invariant (it sees a bag of words without inherent order), I need to inject order information.
I add **Positional Embeddings** to the token embeddings. These encode the position of each token in the sequence. 
*   **Sequence Length**: Fixed to `512` tokens in my configuration.

## Step 4: The Embedding Block
My Embedding Block combines the previous two steps:
1.  Look up Token Embeddings.
2.  Look up (or compute) Positional Embeddings.
3.  **Add** them together.
4.  Apply `Dropout` for regularization.

## Step 5: Self-Attention Block (The Core)
This is the heart of the Transformer. It allows tokens to "look at" other tokens in the sequence to gather context.

**The Logic:**
-   **Input**: Takes the output from the Embedding Block (or previous layer).
-   **Q, K, V**: It projects the input into Query (Q), Key (K), and Value (V) matrices of shape `[B, S, D]` (Batch, Seq, Dim).
-   **Heads**: These are split into multiple heads. This allows the model to focus on different types of relationships simultaneously (e.g., one head for grammar, one for rhyme).
-   **Scaling**: I scale Q and K by `1/sqrt(head_dim)`. This prevents dot products from growing too large, which would otherwise push Softmax into regions with extremely small gradients.

**Forward Pass Implementation:**
1.  Reshape Q, K, V into multi-head format: `[B, H, S, head_dim]`.
2.  **Flash Attention**: If enabled, I use optimized kernels to compute attention scores.
3.  **Manual Attention** (fallback): `softmax((Q @ K.T) / scale) @ V`.
4.  **Merge**: The heads are concatenated back to `[B, S, D]`.
5.  **Projection**: Passed through a final linear output projection.

## Step 6: Feed Forward Block
Each token is processed independently by this block to extract deeper features.
-   It takes the output of the Self-Attention block.
-   **Structure**: `Linear` (expansion) → `GELU` (activation) → `Dropout` → `Linear` (projection).
-   It creates a "memory" capacity for the model independent of interactions with other tokens.

## Step 7: The Transformer Block
I combine the pieces into a single repeatable unit (Pre-Norm architecture):

1.  **Input** $x$
2.  **Attention Branch**: $x + \text{SelfAttention}(\text{LayerNorm}(x))$
3.  **FeedForward Branch**: $y + \text{FeedForward}(\text{LayerNorm}(y))$

The use of **Residual Connections** (adding the input to the output) is crucial for gradient flow in deep networks.

## Step 8: Building the PrimeGPT Model
Finally, I assemble the full model:

1.  **Input Pipeline**: Token Indices → Embedding Block.
2.  **The Stack**: Pass the embeddings through $N$ stacked Transformer Blocks.
3.  **Final Norm**: Apply a final `LayerNorm` to stabilize the output features.
4.  **Language Head**: A linear layer that projects the final embeddings back to the Vocabulary size (logits).
5.  **Weight Tying**: I tie the weights of this final head with the initial Token Embeddings to save parameters and improve coherence.
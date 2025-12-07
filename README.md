# Prime_LLM
LLM trained from scratch for generating music lyrics

The entire process is as follows:

Step 1: Tokenizer
This converts the words to integers. Basically based on mapping of words to integers. Each word is mapped to a unique integer. 

Step 2: Token Embeddings
This converts the integer array to a continuous vector for each token. Each token has its unique vector representation of n-dimension, this is so that the model can learn the semantic relationships between the tokens. For example:
- love = [0.89, 0.22, -0.14, ...]
- adore = [0.91, 0.18, -0.11, ...]
These two vectors are close in euclidean space, close in cosine similarity, have similar direction and thats how the LLM knows that these two words are close in semantic meaning.

Step 3: Positional Embeddings
This converts the input_ids to positional embeddings, by encodeing the positional information of each token in the sequence. The sequence length is same as batch size which is fixed to 512 tokens in this case.

Step 4: Embedding Block
The embedding block simply adds the token and the positional embeddings together and applies dropout to it and then returns the output.

Step 5: Self Attention

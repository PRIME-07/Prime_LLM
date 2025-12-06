import pandas as pd

print("Loading cleaned dataset...")
df = pd.read_csv("data/cleaned_lyrics_dataset.csv")

print("\nCreating tokenizer data...")
with open("data/lyrics_corpus.txt", "w", encoding = "utf-8") as f:
    for line in df["lyrics"]:
        f.write(str(line).strip() + "\n")

print("\nTokenizer data created!")
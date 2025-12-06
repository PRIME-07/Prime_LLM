# Data Cleaning and Preprocessing

import pandas as pd
import re

print("Loading dataset...")
df = pd.read_csv("data/lyrics_dataset.csv")

print("Columns from original dataset:", df.columns)

# Keep only genre (tag) and lyrics
print("\nDropping irrelevant columns...")
df = df[["tag", "lyrics"]]
print("Columns after cleaning:", df.columns)

print("\nDropping rows with missing values...")
df = df.dropna()

# Cleaning functions using regex
print("\nCleaning lyrics...")
def clean_metadata(text):
    # Lowercase for consistency
    text = text.lower()
    
    # Remove "Produced by", "Written by", etc.
    text = re.sub(r"\[.*?by.*?\]", "", text)
    
    # Remove section headers with artist names: "[verse 1: jay-z]"
    text = re.sub(r"\[(verse|chorus|bridge|intro|outro|hook).*?\]", 
                  lambda m: f"[{m.group(1)}]", 
                  text)
    
    # Normalize labels (remove numbers)
    text = re.sub(r"\[(verse|chorus|bridge|intro|outro|hook)\s*\d*\]", r"[\1]", text)
    
    # Remove other noise (sample, annotations, credits)
    text = re.sub(r"\[(spoken|skit|interlude|breakdown|sample|instrumental).*?\]", "", text)

    # Remove repeated newlines/spaces
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def add_genre_token(row):
    genre = row["tag"].lower().strip()

    # normalize genre tokens
    genre_token = f"<genre_{genre}>"
    return genre_token + "\n" + row["lyrics"]

df["lyrics"] = df["lyrics"].apply(clean_metadata)

# Add newlines between sections
df["lyrics"] = df["lyrics"].str.replace(r"\] *<", "]\n<", regex=True)

print("\nAdding genre tokens...")
df["lyrics"] = df.apply(add_genre_token, axis=1)

print("\nSaving cleaned dataset...")
df[["lyrics"]].to_csv("data/cleaned_lyrics_dataset.csv", index=False)
print("\nDone! Final shape:", df.shape)


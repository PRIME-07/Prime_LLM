import torch
from torch.utils.data import Dataset
import pandas as pd
import sentencepiece as spm
import random


class LyricsDataset(Dataset):
    def __init__(self, csv_path, tokenizer_path, seq_len=512):
        self.seq_len = seq_len

        # Load tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)

        # IDs for special tokens
        self.prompt_id = self.sp.piece_to_id("<prompt>")
        self.song_start_id = self.sp.piece_to_id("<song>")
        self.song_end_id = self.sp.piece_to_id("</song>")

        # Load dataset
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["lyrics"])
        df["lyrics"] = df["lyrics"].astype(str)

        print("Unique lyrics types:", df["lyrics"].apply(type).unique())

        # Helper: filter non-ASCII text
        def is_acceptable(text: str) -> bool:
            # drops emojis, Japanese, Cyrillic, etc.
            return text.isascii()

        # Tokenize all lyrics into one long stream
        all_tokens = []
        skipped_non_ascii = 0
        skipped_encode_fail = 0

        genre_tokens = [
            "<genre_pop>",
            "<genre_rap>",
            "<genre_rock>",
            "<genre_rb>",
        ]

        emotion_tokens = [
            "<happy>",
            "<sad>",
            "<romantic>",
            "<angry>",
        ]

        instruction_templates = [
            "write a song about weekends",
            "write a song about love",
            "write a song about missing home",
            "write a song about freedom",
            "write a song about summer nights",
        ]

        PROMPT_DROPOUT_PROB = 0.15

        for line in df["lyrics"]:
            if not is_acceptable(line):
                skipped_non_ascii += 1
                continue

            genre = random.choice(genre_tokens)
            emotion = random.choice(emotion_tokens)
            instruction = random.choice(instruction_templates)

            use_prompt = random.random() > PROMPT_DROPOUT_PROB

            if use_prompt:
                training_text = (
                    f"<prompt> {genre} {emotion} {instruction}\n"
                    f"<song>\n"
                    f"{line}\n"
                    f"</song>"
                )
            else:
                training_text = (
                    f"<song>\n"
                    f"{line}\n"
                    f"</song>"
                )

            try:
                tokens = self.sp.encode(training_text, out_type=int)
            except Exception:
                skipped_encode_fail += 1
                continue

            # Append EOS token after each song
            all_tokens.extend(tokens)
            all_tokens.append(self.sp.eos_id())

        if len(all_tokens) == 0:
            raise RuntimeError("No valid tokens found. Check tokenizer and filters.")

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)

        print(f"Total tokens: {len(self.tokens)}")
        print(f"Skipped non-ASCII lyrics: {skipped_non_ascii}")
        print(f"Skipped encoding failures: {skipped_encode_fail}")

    def __len__(self):
        # Number of possible (x, y) windows
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]

        # create loss mask for instruction tuning
        mask = torch.ones_like(y)

        song_pos = (x == self.song_start_id).nonzero(as_tuple=True)
        if len(song_pos[0]) > 0:
            song_pos = song_pos[0][0]

            # mask everything up to and including <song>
            mask[: song_pos + 1] = 0

            # also mask the target immediately after <song>
            mask[song_pos] = 0

            positions = torch.arange(len(x), device=x.device)

            # additionally mask instruction tokens that appear after <song>
            instruction_ids = [
                self.sp.piece_to_id("<prompt>"),
                self.sp.piece_to_id("<genre_pop>"),
                self.sp.piece_to_id("<genre_rap>"),
                self.sp.piece_to_id("<genre_rock>"),
                self.sp.piece_to_id("<genre_rb>"),
                self.sp.piece_to_id("<happy>"),
                self.sp.piece_to_id("<sad>"),
                self.sp.piece_to_id("<romantic>"),
                self.sp.piece_to_id("<angry>"),
            ]

            for tid in instruction_ids:
                mask[(x == tid) & (positions > song_pos)] = 0

        return x, y, mask


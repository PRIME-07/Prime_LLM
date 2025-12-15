import os
import sys
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
import sentencepiece as spm
import random

# Allow importing from model folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
sys.path.append(MODEL_DIR)

from config import Config
from dataset import LyricsDataset
from prime_gpt import PrimeGPT
from dataloader import create_dataloader

# Training hyperparameters
BATCH_SIZE = 12
GRAD_ACCUM_STEPS = 12
MAX_STEPS = 200_000
LOG_EVERY = 100
SAVE_EVERY = 2000
SAMPLE_EVERY = 5000
GRAD_CLIP = 1.0
USE_AMP = True

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "samples")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Resume training from checkpoint
resume_path = "./checkpoints/primegpt_step_84000.pt"

# Text generation
@torch.no_grad()
def generate_sample(model, tokenizer, device, step):
    model.eval()

    genres = ["<genre_pop>", "<genre_rap>", "<genre_rock>", "<genre_rb>"]
    emotions = ["<happy>", "<sad>", "<romantic>", "<angry>"]

    genre = random.choice(genres)
    emotion = random.choice(emotions)

    user_prompt = "write a song about weekends"
    prompt_text = f"<prompt> {genre} {emotion} {user_prompt}\n<song>\n"

    input_ids = torch.tensor(
        tokenizer.encode(prompt_text),
        device=device
    ).unsqueeze(0)

    song_end_id = tokenizer.piece_to_id("</song>")
    chorus_id = tokenizer.piece_to_id("[chorus]")

    banned_ids = [
        tokenizer.piece_to_id("<prompt>"),
        tokenizer.piece_to_id("<song>"),
        tokenizer.piece_to_id("<genre_pop>"),
        tokenizer.piece_to_id("<genre_rap>"),
        tokenizer.piece_to_id("<genre_rock>"),
        tokenizer.piece_to_id("<genre_rb>"),
        tokenizer.piece_to_id("<happy>"),
        tokenizer.piece_to_id("<sad>"),
        tokenizer.piece_to_id("<romantic>"),
        tokenizer.piece_to_id("<angry>"),
    ]

    output = model.generate(
        input_ids,
        max_new_tokens=250,
        temperature=0.9,
        top_p=0.9,
        repetition_penalty=1.2,
        eos_token_ids=[song_end_id],
        chorus_token_id=chorus_id,
        banned_token_ids=banned_ids,
    )

    text = tokenizer.decode(output[0].tolist())

    def extract_song(text):
        if "<song>" in text and "</song>" in text:
            return text.split("<song>", 1)[1].split("</song>", 1)[0]
        return text

    song_text = extract_song(text)

    def prompt_adherence(prompt, generation):
        pw = set(prompt.lower().split())
        gw = set(generation.lower().split())
        return len(pw & gw) / max(len(pw), 1)

    score = prompt_adherence(user_prompt, song_text)
    print(f"Prompt adherence score: {score:.2f}")

    path = os.path.join(SAMPLE_DIR, f"step_{step}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Sample saved → {path}")
    model.train()

# Training loop
def train():
    config = Config()
    device = torch.device(config.device)
    print(f"Using device: {device}")

    model = PrimeGPT(config).to(device)
    model.train()

    # Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(PROJECT_ROOT, "data/lyrics_bpe.model"))

    dataset = LyricsDataset(
        csv_path=os.path.join(PROJECT_ROOT, "data/cleaned_lyrics_dataset.csv"),
        tokenizer_path=os.path.join(PROJECT_ROOT, "data/lyrics_bpe.model"),
        seq_len=config.max_seq_len,
    )

    dataloader = create_dataloader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    warmup_steps = config.warmup_steps
    total_steps = MAX_STEPS

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    scaler = GradScaler(device="cuda", enabled=USE_AMP)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    while global_step < total_steps:
        for x, y, mask in dataloader:
            if global_step >= total_steps:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with autocast(device_type="cuda", enabled=USE_AMP):
                logits = model(x)
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                )
                loss = (loss * mask.view(-1)).sum() / mask.sum()
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (global_step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            global_step += 1

            if global_step % LOG_EVERY == 0:
                print(
                    f"step {global_step:6d} | "
                    f"loss {loss.item() * GRAD_ACCUM_STEPS:.4f} | "
                    f"lr {scheduler.get_last_lr()[0]:.6f}"
                )

            if global_step % SAVE_EVERY == 0:
                path = os.path.join(
                    CHECKPOINT_DIR,
                    f"primegpt_step_{global_step}.pt"
                )
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "step": global_step,
                    },
                    path
                )
                print(f"Checkpoint saved → {path}")

            if global_step % SAMPLE_EVERY == 0:
                generate_sample(model, sp, device, global_step)

    print("Training complete.")

if __name__ == "__main__":
    train()

# mini_gpt_pdf_trainer.py — Multi-file PDF loader + GPT-style Transformer

import fitz  # PyMuPDF
import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 1. PDF TEXT EXTRACTION (MULTI-FILE)
# -----------------------------
def extract_text_from_pdfs(directory):
    all_text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            print("\nReading from", filename)
            path = os.path.join(directory, filename)
            doc = fitz.open(path)
            for page in doc:
                all_text += page.get_text()
    return all_text

# -----------------------------
# 2. TOKENIZATION + VOCAB
# -----------------------------
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def build_vocab(words, min_freq=1):
    counter = Counter(words)
    vocab = {word for word, freq in counter.items() if freq >= min_freq}
    word2idx = {word: idx+2 for idx, word in enumerate(sorted(vocab))}
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

def encode(words, word2idx):
    return [word2idx.get(word, word2idx['<UNK>']) for word in words]

# -----------------------------
# 3. CUSTOM DATASET
# -----------------------------
class TextDataset(Dataset):
    def __init__(self, encoded, context_len):
        self.data = [(
            torch.tensor(encoded[i:i+context_len]),
            torch.tensor(encoded[i+context_len])
        ) for i in range(len(encoded) - context_len)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# -----------------------------
# 4. MINI GPT-STYLE TRANSFORMER (multi-layer)
# -----------------------------
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, context_len=5, embed_dim=64, num_heads=2, num_layers=2):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, context_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ln = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        tok = self.token_embed(x)
        x = tok + self.pos_embed[:, :x.size(1), :]
        mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(x.device)
        x = self.transformer(x, mask=mask)
        x = self.ln(x[:, -1, :])  # final token only
        return self.output(x)

# -----------------------------
# 5. TRAINING LOOP
# -----------------------------
def train_model(
    model, dataloader, epochs=500, lr=1e-4,
    word2idx=None, idx2word=None,
    seed_tokens=None, context_len=5,
    print_every=10, save_path="best_model.pth",
    patience=20  # stop if no improvement after 20 epochs
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_loss = float('inf')
    epochs_since_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for context, target in dataloader:
            context, target = context.to(device), target.to(device)
            logits = model(context)
            loss = loss_fn(logits, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

        # Save best model
        if total_loss < best_loss:
            best_loss = total_loss
            epochs_since_improvement = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path} (Loss: {best_loss:.4f})")
        else:
            epochs_since_improvement += 1

        # Optional generation
        if word2idx and idx2word and seed_tokens and (epoch % print_every == 0 or epoch == epochs - 1):
            print("\nGenerated (from token IDs):")
            print(generate(model, seed_tokens, idx2word, word2idx, steps=20))
            print("\nGenerated (from seed string):")
            print(generate_from_string(model, "the model could", word2idx, idx2word, context_len, steps=20))
            print("-" * 60)

        # Early stopping condition
        if epochs_since_improvement >= patience:
            print(f"No improvement for {patience} epochs. Stopping early.")
            break

# -----------------------------
# 6. TOP-K SAMPLING
# -----------------------------
def sample_top_k(probs, k=10):
    topk = torch.topk(probs, k)
    probs_topk = F.softmax(topk.values, dim=-1)
    idx = torch.multinomial(probs_topk, num_samples=1).item()
    return topk.indices[idx].item()

# -----------------------------
# 7. GENERATION FROM TOKENS
# -----------------------------
def generate(model, seed, idx2word, word2idx, steps=20, temperature=1.0, top_k=10):
    model.eval()
    device = next(model.parameters()).device
    seed = seed[-5:]
    result = []

    for _ in range(steps):
        x = torch.tensor(seed).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)[0] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = sample_top_k(probs, k=top_k)

        result.append(idx2word.get(next_token, "<?>"))
        seed = seed[1:] + [next_token]

    return ' '.join(result)

# -----------------------------
# 8. GENERATION FROM STRING SEED
# -----------------------------
def generate_from_string(model, seed_text, word2idx, idx2word, context_len=5, steps=20, temperature=1.0, top_k=10):
    words = tokenize(seed_text)
    tokens = [word2idx.get(w, word2idx['<UNK>']) for w in words]
    if len(tokens) < context_len:
        tokens = [word2idx['<PAD>']] * (context_len - len(tokens)) + tokens
    return generate(model, tokens, idx2word, word2idx, steps, temperature, top_k)

# -----------------------------
# 9. MAIN SCRIPT
# -----------------------------
if __name__ == "__main__":
    save_path="best_model.pth"
    text = extract_text_from_pdfs("Training")
    words = tokenize(text)
    word2idx, idx2word = build_vocab(words)

     # Save vocab to file for later use in inference
    torch.save({
        "word2idx": word2idx,
        "idx2word": idx2word
    }, "vocab.pt")
    print("✅ Vocab saved to vocab.pt")
    
    encoded = encode(words, word2idx)

    context_len = 10
    dataset = TextDataset(encoded, context_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    vocab_size=len(word2idx)
    print(f"Vocab size: {vocab_size}, Dataset size: {len(dataset)}")

    
    model = MiniGPT(vocab_size, context_len=context_len, num_layers=2)
    train_model(
        model, dataloader,
        epochs=500, lr=1e-4,
        word2idx=word2idx, idx2word=idx2word,
        seed_tokens=encoded[:context_len],
        context_len=context_len,
        print_every=10,
        save_path=save_path,
        patience=20
    ) 

    print("\nLoading model for generation...")
    model.load_state_dict(torch.load(save_path))
    model.eval()

    print("\nGenerated (from token IDs):")
    seed = encoded[:context_len]
    print(generate(model, seed, idx2word, word2idx, top_k=10))

    seed_string = "the model could"
    print("\nGenerated (from seed string):",seed_string)
    print(generate_from_string(model, seed_string, word2idx, idx2word, top_k=10))

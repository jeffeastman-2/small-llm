import torch
import torch.nn as nn
import torch.nn.functional as F
import re

# ------------------------------------------------
# MODEL CLASS (should match your training model)
# ------------------------------------------------
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

# ------------------------------------------------
# GENERATION FUNCTION
# ------------------------------------------------
def generate_from_string(model, seed_str, word2idx, idx2word, context_len=5, steps=20):
    words = re.findall(r'\b\w+\b', seed_str.lower())
    tokens = [word2idx.get(w, word2idx['<UNK>']) for w in words][-context_len:]
    return generate(model, tokens, idx2word, word2idx, steps)

def generate(model, seed, idx2word, word2idx, steps=20, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device
    seed = seed[-5:]
    result = []

    for _ in range(steps):
        x = torch.tensor(seed).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)[0] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        result.append(idx2word.get(next_token, "<UNK>"))
        seed = seed[1:] + [next_token]

    return ' '.join(result)

# ------------------------------------------------
# LOAD VOCAB
# ------------------------------------------------
# If you saved it before: torch.save({"word2idx": ..., "idx2word": ...}, "vocab.pt")
vocab_data = torch.load("use_vocab.pt")
word2idx = vocab_data["word2idx"]
idx2word = vocab_data["idx2word"]

# ------------------------------------------------
# LOAD MODEL & GENERATE
# ------------------------------------------------
context_len = 10
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

model = MiniGPT(vocab_size=len(word2idx), context_len=context_len)
model.load_state_dict(torch.load("use_model.pth", map_location=device))
model = model.to(device)

while True:
    prompt = input("Prompt> ")
    if not prompt:
        break
    print(generate_from_string(model, prompt, word2idx,idx2word, context_len, steps=20))

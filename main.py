import os, math, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sentencepiece as spm   # solo para tokenización

# ------------------------------------------------------------------
# 1.  Tokenizador
# ------------------------------------------------------------------
class BPETokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.vocab_size = self.sp.GetPieceSize()

    def encode(self, text: str) -> list[int]:
        return self.sp.EncodeAsIds(text)

    def decode(self, ids: list[int]) -> str:
        return self.sp.DecodeIds(ids)


# ------------------------------------------------------------------
# 2.  Dataset de lenguaje
# ------------------------------------------------------------------
class TextDataset(Dataset):
    """
    El dataset devuelve pares (input_ids, labels) donde
    input_ids[i] = token t_i          y
    labels[i]   = token t_{i+1}
    Se recorta en bloques de `block_size`.
    """
    def __init__(self, texts: list[str], tokenizer: BPETokenizer,
                 block_size: int = 128):
        self.block_size = block_size
        self.ids = []
        for txt in texts:
            self.ids.extend(tokenizer.encode(txt))
        # Añadimos un token EOS (id = vocab-1) al final de cada texto
        self.ids.append(tokenizer.vocab_size - 1)

    def __len__(self):
        return len(self.ids) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end   = start + self.block_size
        x = torch.tensor(self.ids[start:end], dtype=torch.long)
        y = torch.tensor(self.ids[start+1:end+1], dtype=torch.long)
        return {"input_ids": x, "labels": y}


# ------------------------------------------------------------------
# 3.  Modelo Transformer “desde cero”
# ------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads

        # Proyecciones lineales para Q, K, V y salida final
        self.qkv_proj   = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj   = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)          # cada uno: (B,T,H,D)

        # Escalar dot‑product attention
        attn_scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output  = torch.einsum("bhts,bshd->bthd", attn_weights, v)

        # Concatenar cabezas y proyección final
        attn_out = attn_output.reshape(B, T, C)
        return self.out_proj(attn_out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_dim: int, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff  = FeedForward(d_model, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Self‑attention + residual
        attn_out = self.attn(self.ln1(x))
        x = x + self.dropout1(attn_out)

        # FFN + residual
        ff_out  = self.ff(self.ln2(x))
        return x + self.dropout2(ff_out)


class GPTLikeLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int   = 128,
                 num_heads: int = 8,
                 hidden_dim: int=256,
                 num_layers: int=4,
			     max_seq_len: int = 512,
                 dropout: float   = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb    = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.blocks     = nn.ModuleList([
            TransformerBlock(d_model, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final   = nn.LayerNorm(d_model)
        self.head       = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids : (B, T)  dtype=torch.long
        returns logits : (B, T, vocab)
        """
        B, T = input_ids.size()
        x = self.token_emb(input_ids) + self.pos_emb[:, :T, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_final(x)
        return self.head(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

# ------------------------------------------------------------------
# 4.  Entrenamiento / Evaluación
# ------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0.0
    for batch in loader:
        inp   = batch["input_ids"].to(device)
        target= batch["labels"   ].to(device)

        logits = model(inp)                     # (B,T,V)
        loss   = loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * inp.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            inp   = batch["input_ids"].to(device)
            target= batch["labels"   ].to(device)

            logits = model(inp)
            loss   = loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))
            total_loss += loss.item() * inp.size(0)
    return total_loss / len(loader.dataset)


# ------------------------------------------------------------------
# 5.  Generación de texto
# ------------------------------------------------------------------
def generate(model: nn.Module,
             tokenizer: BPETokenizer,
             prompt: str,
             max_new_tokens: int = 50,
             temperature: float = 1.0,
             top_k: int = 200,
             device='cpu') -> str:
    """
    Prompt → token IDs → autoregressive generation → string
    """
    model.eval()
    inp_ids = tokenizer.encode(prompt)
    generated = inp_ids.copy()

    for _ in range(max_new_tokens):
        cur_input = torch.tensor([generated], dtype=torch.long, device=device)  # (1,T)
        logits = model(cur_input)[0, -1] / temperature                     # (V,)
        if top_k > 0:
            v, ix = torch.topk(logits, top_k)
            probs = torch.softmax(v, dim=-1)
            idx   = torch.multinomial(probs, num_samples=1).item()
            next_token = int(ix[idx].item())
        else:
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

        generated.append(next_token)

        # Stop si llega al EOS (id = vocab-1) o a un token de nueva línea
        if next_token == tokenizer.vocab_size - 1 or next_token == tokenizer.sp.PieceToId('\n'):
            break

    return tokenizer.decode(generated)


# ------------------------------------------------------------------
# 6.  Main – Entrenamiento mínimo
# ------------------------------------------------------------------
def main():
    # ---------- Parámetros ----------
    BATCH_SIZE   = 32
    NUM_EPOCHS   = 25
    LR           = 3e-4
    BLOCK_SIZE   = 128
    MAX_SEQ_LEN  = 512

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # ---------- Tokenizador ----------
    tokenizer = BPETokenizer('bpe_sp.model')
    vocab_size = tokenizer.vocab_size

    # ---------- Datos ----------
    # Supón que tienes un directorio `data/` con archivos `.txt`
    data_dir   = Path('./data')
    texts      = [p.read_text(encoding='utf-8') for p in data_dir.glob('*.txt')]
    dataset    = TextDataset(texts, tokenizer, block_size=BLOCK_SIZE)
    loader     = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            drop_last=True)

    # ---------- Modelo ----------
    model = GPTLikeLM(vocab_size=vocab_size,
                      d_model=128,
                      num_heads=8,
                      hidden_dim=256,
                      num_layers=4,
                      max_seq_len=MAX_SEQ_LEN).to(device)

    print(f"Parámetros del modelo: {model.param_count():,}")

    # ---------- Optimización ----------
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=-100)  # no se usa

    # ---------- Loop de entrenamiento ----------
    for epoch in range(1, NUM_EPOCHS+1):
        train_loss = train_one_epoch(model, loader, optimizer, device, loss_fn)
        val_loss   = evaluate(model, loader, device, loss_fn)

        print(f"Epoch {epoch:02d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

    # ---------- Guardar modelo ----------
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_path'   : 'bpe_sp.model',
        'config'           : {
            'vocab_size'  : vocab_size,
            'd_model'     : 128,
            'num_heads'   : 8,
            'hidden_dim'  : 256,
            'num_layers'  : 4,
            'max_seq_len' : MAX_SEQ_LEN
        }
    }, 'llm_es.pth')

    # ---------- Prueba de generación ----------
    prompt = "Había una vez "
    print("\n=== Generación ===")
    print(generate(model, tokenizer, prompt, max_new_tokens=80,
                   temperature=0.7, top_k=200, device=device))


if __name__ == "__main__":
    main()

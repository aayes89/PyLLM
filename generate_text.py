import torch
import sentencepiece as spm
from pathlib import Path

# Clases necesarias (sin cambios, asumiendo que son las mismas del modelo entrenado)
class BPETokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.vocab_size = self.sp.GetPieceSize()

    def encode(self, text: str) -> list[int]:
        return self.sp.EncodeAsIds(text)

    def decode(self, ids: list[int]) -> str:
        return self.sp.DecodeIds(ids)

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = torch.nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        attn_scores = torch.einsum("bthd,bshd->bhts", q, k) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.einsum("bhts,bshd->bthd", attn_weights, v)
        attn_out = attn_output.reshape(B, T, C)
        return self.out_proj(attn_out)

class FeedForward(torch.nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_model, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_dim: int, dropout=0.1):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, hidden_dim)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attn(self.ln1(x))
        x = x + self.dropout1(attn_out)
        ff_out = self.ff(self.ln2(x))
        return x + self.dropout2(ff_out)

class GPTLikeLM(torch.nn.Module):
    def __init__(self, vocab_size: int, d_model: int=128, num_heads: int=8,
                 hidden_dim: int=256, num_layers: int=4, max_seq_len: int=512,
                 dropout: float=0.1):
        super().__init__()
        self.token_emb = torch.nn.Embedding(vocab_size, d_model)
        self.pos_emb = torch.nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(d_model, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.size()
        x = self.token_emb(input_ids) + self.pos_emb[:, :T, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_final(x)
        return self.head(x)

def generate(model: torch.nn.Module, tokenizer: BPETokenizer, prompt: str,
             max_new_tokens: int=80, temperature: float=0.7, top_k: int=50,
             device='cpu') -> str:
    model.eval()
    inp_ids = tokenizer.encode(prompt)
    # Validar que los IDs estén dentro del rango del vocab_size del modelo
    max_id = max(inp_ids)
    if max_id >= tokenizer.vocab_size:
        raise ValueError(f"ID de token {max_id} excede el vocab_size del tokenizador ({tokenizer.vocab_size})")
    generated = inp_ids.copy()

    for _ in range(max_new_tokens):
        cur_input = torch.tensor([generated], dtype=torch.long, device=device)
        logits = model(cur_input)[0, -1] / temperature
        if top_k > 0:
            v, ix = torch.topk(logits, min(top_k, logits.size(-1)))
            probs = torch.softmax(v, dim=-1)
            idx = torch.multinomial(probs, num_samples=1).item()
            next_token = int(ix[idx].item())
        else:
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
        generated.append(next_token)
        if next_token == tokenizer.vocab_size - 1 or next_token == tokenizer.sp.PieceToId('\n'):
            break
    return tokenizer.decode(generated)

def load_model(checkpoint_path: str, device: str='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    model = GPTLikeLM(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    tokenizer = BPETokenizer(checkpoint['tokenizer_path'])
    # Validar compatibilidad entre modelo y tokenizador
    if config['vocab_size'] != tokenizer.vocab_size:
        raise ValueError(f"El vocab_size del modelo ({config['vocab_size']}) no coincide con el del tokenizador ({tokenizer.vocab_size})")
    return model, tokenizer

def main():
    device = torch.device('cpu')
    checkpoint_path = 'llm_es.pth'
    try:
        model, tokenizer = load_model(checkpoint_path, device)
        print(f"Modelo cargado con {sum(p.numel() for p in model.parameters()):,} parámetros")
        prompt = "Había una vez "
        print("\n=== Generación ===")
        print(generate(model, tokenizer, prompt, max_new_tokens=80,
                       temperature=0.7, top_k=50, device=device))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
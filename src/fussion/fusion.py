"""FusionLLM — cross-attention fusion layers inside a frozen LLM."""
import math, time
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class CrossAttentionLayer(nn.Module):
    """Cross-attention: query from LLM hidden, key/value from encoder tokens."""
    def __init__(self, d_model, nhead=4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)

    def forward(self, x, visual_tokens):
        residual = x
        x = self.norm(x)
        x = self.attn(x, visual_tokens, visual_tokens)[0]
        return residual + x


class WrappedLayer(nn.Module):
    """Wraps an LLM layer and adds cross-attention after it."""
    def __init__(self, original_layer, cross_attn=None):
        super().__init__()
        self.original_layer = original_layer
        self.cross_attn = cross_attn

    def forward(self, *args, **kwargs):
        x = self.original_layer(*args, **kwargs)
        if isinstance(x, tuple):
            hidden, rest = x[0], x[1:]
        else:
            hidden, rest = x, ()
        if self.cross_attn is not None and hasattr(self, '_visual_tokens'):
            hidden = self.cross_attn(hidden, self._visual_tokens)
        return (hidden,) + rest if rest else hidden


class FusionLLM(nn.Module):
    """Frozen LLM with inserted cross-attention fusion layers every K layers."""

    def __init__(self, llm_name="gpt2", encoder_dim=512, every_k_layers=4,
                 nhead=4, device=None, verbose=False):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.every_k = every_k_layers
        self.verbose = verbose

        self.tok = AutoTokenizer.from_pretrained(llm_name)
        self.tok.pad_token = self.tok.eos_token or self.tok.pad_token

        self.llm = AutoModelForCausalLM.from_pretrained(llm_name).to(self.device)
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad = False

        cfg = self.llm.config
        self.d_model = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or 768
        self.llm_name = llm_name
        self.vocab_size = cfg.vocab_size

        self.visual_proj = nn.Linear(encoder_dim, self.d_model, bias=False)
        nn.init.zeros_(self.visual_proj.weight)

        layers = self._get_layers()
        self.wrapped_layers = nn.ModuleList()
        for i in range(len(layers)):
            ca = CrossAttentionLayer(self.d_model, nhead) if i % every_k_layers == 0 else None
            wl = WrappedLayer(layers[i], ca)
            self.wrapped_layers.append(wl)
            layers[i] = wl

        total_params = sum(p.numel() for p in self.visual_proj.parameters())
        for wl in self.wrapped_layers:
            if wl.cross_attn is not None:
                total_params += sum(p.numel() for p in wl.cross_attn.parameters())
        n_fusion = sum(1 for wl in self.wrapped_layers if wl.cross_attn is not None)
        if verbose:
            print(f"  Fusion layers: {n_fusion} x CrossAttention (every {every_k_layers})")
            print(f"  Trainable params: {total_params:,}")

    def get_trainable_params(self):
        params = list(self.visual_proj.parameters())
        for wl in self.wrapped_layers:
            if wl.cross_attn is not None:
                params.extend(wl.cross_attn.parameters())
        return params

    def _get_layers(self):
        if hasattr(self.llm, "transformer") and hasattr(self.llm.transformer, "h"):
            return self.llm.transformer.h
        elif hasattr(self.llm, "model") and hasattr(self.llm.model, "layers"):
            return self.llm.model.layers
        raise AttributeError(f"Cannot find layers in {type(self.llm)}")

    def forward(self, input_ids, visual_tokens, labels=None, attention_mask=None):
        B, T = input_ids.shape
        N = visual_tokens.shape[1]
        llm_dtype = next(self.llm.parameters()).dtype
        visual = self.visual_proj(visual_tokens).to(llm_dtype)
        embeds = self.llm.get_input_embeddings()(input_ids)
        combined = torch.cat([visual, embeds], dim=1)
        vis_mask = torch.ones(B, N, dtype=torch.long, device=self.device)
        txt_mask = attention_mask if attention_mask is not None else torch.ones(B, T, dtype=torch.long, device=self.device)
        full_mask = torch.cat([vis_mask, txt_mask], dim=1)

        for wl in self.wrapped_layers:
            if wl.cross_attn is not None:
                wl._visual_tokens = visual

        out = self.llm(inputs_embeds=combined, attention_mask=full_mask, return_dict=True)

        for wl in self.wrapped_layers:
            if hasattr(wl, '_visual_tokens'):
                del wl._visual_tokens

        logits = out.logits
        if labels is not None:
            loss = F.cross_entropy(logits[:, N:, :].reshape(-1, self.vocab_size), labels.reshape(-1), ignore_index=-100)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

    @torch.no_grad()
    def generate(self, visual_tokens, prompt="", max_new=50, temperature=0.8,
                 top_p=0.9, repetition_penalty=1.0):
        llm_dtype = next(self.llm.parameters()).dtype
        visual = self.visual_proj(visual_tokens).to(llm_dtype)

        for wl in self.wrapped_layers:
            if wl.cross_attn is not None:
                wl._visual_tokens = visual

        prompt_ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)
        curr = torch.cat([visual, prompt_embeds], dim=1)
        gen_ids, past, eos_id = [], None, self.tok.eos_token_id

        for _ in range(max_new):
            inp = curr if past is None else curr[:, -1:, :]
            out = self.llm(inputs_embeds=inp, past_key_values=past, use_cache=True)
            logits = out.logits[:, -1, :]
            if temperature == 0:
                next_id = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if repetition_penalty != 1.0 and gen_ids:
                    for gid in set(gen_ids):
                        logits[:, gid] /= repetition_penalty
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cum_probs > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = 0
                    logits[remove.scatter(1, sorted_indices, remove)] = float('-inf')
                next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
            if next_id.item() == eos_id: break
            gen_ids.append(next_id.item())
            curr = self.llm.get_input_embeddings()(next_id)
            past = out.past_key_values

        for wl in self.wrapped_layers:
            if hasattr(wl, '_visual_tokens'):
                del wl._visual_tokens

        text = self.tok.decode(gen_ids) if gen_ids else "(empty)"
        for t in ("<|user|>", "<|assistant|>", "</s>", "<s>", "<pad>"):
            text = text.replace(t, "")
        return text.strip()


def train_fusion(model, encoder, train_src, train_tgt,
                 val_src=None, val_tgt=None,
                 steps=20, lr=3e-4, batch_size=4, max_len=64, max_encoder_len=50,
                 verbose=True):
    if verbose:
        print(f"Training FusionLLM ({steps} steps)...")
    t0 = time.time()
    all_vis = []
    for x in train_src:
        t = encoder([x])
        B, L, D = t.shape
        if L > max_encoder_len: t = t[:, :max_encoder_len, :]
        elif L < max_encoder_len: t = torch.cat([t, torch.zeros(B, max_encoder_len-L, D)], dim=1)
        all_vis.append(t)
    all_vis = torch.cat(all_vis, dim=0)

    opt = torch.optim.AdamW(model.get_trainable_params(), lr=lr, weight_decay=0.01)
    for step in range(steps):
        model.train()
        opt.zero_grad()
        total_loss = 0
        for i in range(0, len(train_src), batch_size):
            j = min(i + batch_size, len(train_src))
            vis = all_vis[i:j].to(model.device)
            enc = model.tok(train_tgt[i:j], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            ids = enc.input_ids.to(model.device)
            mask = enc.attention_mask.to(model.device)
            labels = ids.clone()
            labels[mask == 0] = -100
            out = model(ids[:, :-1], vis, labels=labels[:, 1:], attention_mask=mask[:, :-1])
            loss = out["loss"]
            loss.backward()
            total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), 1.0)
        opt.step()
        if verbose and (step + 1) % 5 == 0:
            print(f"  Step {step+1}/{steps} loss={total_loss/max(len(train_src)//batch_size,1):.4f}")
    if verbose:
        print(f"  Done in {time.time()-t0:.0f}s")

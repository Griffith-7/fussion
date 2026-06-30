"""CrossModalMerger — external bridge between encoder and frozen LLM."""
import math, os, json, copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .bridge import get_bridge
from .encoders import get_encoder


class CrossModalMerger:
    def __init__(self, source_encoder="clip", target_llm="gpt2",
                 bridge_type="mlp", device=None, cache_dir=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bridge_type = bridge_type
        self.cache_dir = cache_dir
        self.source_encoder = get_encoder(source_encoder) if isinstance(source_encoder, str) else source_encoder
        self._setup_llm(target_llm)
        self._setup_bridge()

    def _setup_llm(self, model_name):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
        self.tok.pad_token = self.tok.eos_token or self.tok.pad_token
        self.llm = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=self.cache_dir).to(self.device)
        self.llm.eval()
        cfg = self.llm.config
        self.llm_dim = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or 768
        self.llm_name = model_name

    def _setup_bridge(self):
        self.bridge = get_bridge(self.bridge_type, self.source_encoder.dim, self.llm_dim).to(self.device)

    def encode_source(self, inputs, batch_size=64, verbose=True):
        return self.source_encoder(inputs)

    def _pad_tokens(self, tokens, max_len):
        B, L, D = tokens.shape
        if L >= max_len:
            return tokens[:, :max_len, :]
        pad = torch.zeros(B, max_len - L, D, device=tokens.device, dtype=tokens.dtype)
        return torch.cat([tokens, pad], dim=1)

    def train_bridge(self, source_inputs, target_texts, val_source=None, val_targets=None,
                     steps=20, lr=3e-4, weight_decay=0.01, max_length=64, batch_size=32,
                     eval_every=5, max_encoder_len=50, verbose=True):
        t0 = time.time()
        all_tokens = torch.cat([self._pad_tokens(self.encode_source([x]), max_encoder_len) for x in source_inputs], dim=0)
        n_prefix = all_tokens.shape[1]
        opt = torch.optim.AdamW(self.bridge.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        self.bridge.train()
        best_state, best_loss = None, float('inf')

        for step in range(steps):
            opt.zero_grad()
            total_loss, n_batches = 0.0, 0
            for i in range(0, len(source_inputs), batch_size):
                j = min(i + batch_size, len(source_inputs))
                tokens = all_tokens[i:j].to(self.device)
                prefix = self.bridge(tokens)
                enc = self.tok(target_texts[i:j], return_tensors="pt", padding=True,
                               truncation=True, max_length=max_length)
                ids = enc.input_ids.to(self.device)
                mask = enc.attention_mask.to(self.device)
                embeds = self.llm.get_input_embeddings()(ids)
                llm_dtype = next(self.llm.parameters()).dtype
                combined = torch.cat([prefix.to(llm_dtype), embeds[:, :-1, :]], dim=1)
                ext_mask = torch.cat([torch.ones(j-i, n_prefix, dtype=mask.dtype, device=self.device), mask[:, :-1]], dim=1)
                out = self.llm(inputs_embeds=combined, attention_mask=ext_mask)
                logits = out.logits
                sl = logits[:, n_prefix:, :].contiguous()
                sl_labels = ids[:, 1:].contiguous()
                sl_mask = mask[:, 1:].contiguous()
                loss = F.cross_entropy(sl.view(-1, sl.size(-1)), sl_labels.view(-1), ignore_index=-100, reduction='none')
                loss = (loss.view(sl.shape[0], -1) * sl_mask).sum() / sl_mask.sum()
                total_loss += loss.item()
                n_batches += 1
                loss.backward()
            torch.nn.utils.clip_grad_norm_(self.bridge.parameters(), 1.0)
            opt.step(); sched.step()
            avg_loss = total_loss / max(n_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(self.bridge.state_dict())
            if (step + 1) % eval_every == 0 or step == 0:
                if verbose:
                    msg = f"  Step {step+1}/{steps} loss={avg_loss:.4f}"
                    if val_source is not None:
                        msg += f" val_ppl={self.evaluate(val_source, val_targets):.1f}"
                    print(msg)
        if best_state is not None:
            self.bridge.load_state_dict(best_state)
        self.bridge.eval()
        final_ppl = self.evaluate(val_source, val_targets) if val_source is not None else float('inf')
        if verbose:
            print(f"  Done in {time.time()-t0:.0f}s | Best loss: {best_loss:.4f} | Val PPL: {final_ppl:.1f}")
        self.bridge.val_ppl = final_ppl
        return self.bridge

    @torch.no_grad()
    def evaluate(self, source_inputs, target_texts, max_length=64, max_encoder_len=50):
        if not source_inputs: return float('inf')
        all_tokens = torch.cat([self._pad_tokens(self.encode_source([x]), max_encoder_len) for x in source_inputs], dim=0)
        n_prefix = all_tokens.shape[1]
        self.bridge.eval()
        total_loss = 0.0
        for i in range(len(source_inputs)):
            tokens = all_tokens[i:i+1].to(self.device)
            prefix = self.bridge(tokens)
            enc = self.tok([target_texts[i]], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            ids = enc.input_ids.to(self.device)
            mask = enc.attention_mask.to(self.device)
            combined = torch.cat([prefix.to(next(self.llm.parameters()).dtype), self.llm.get_input_embeddings()(ids)[:, :-1, :]], dim=1)
            ext_mask = torch.cat([torch.ones(1, n_prefix, dtype=mask.dtype, device=self.device), mask[:, :-1]], dim=1)
            out = self.llm(inputs_embeds=combined, attention_mask=ext_mask)
            logits = out.logits
            sl = logits[:, n_prefix:, :].contiguous()
            sl_labels = ids[:, 1:].contiguous()
            sl_mask = mask[:, 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), sl_labels.view(-1), ignore_index=-100, reduction='none')
            loss = (loss.view(sl.shape[0], -1) * sl_mask).sum() / sl_mask.sum()
            total_loss += loss.item()
        return math.exp(total_loss / len(source_inputs))

    def _extract_prefix(self, source_input, max_encoder_len=50):
        tokens = self._pad_tokens(self.encode_source([source_input]), max_encoder_len)
        return self.bridge(tokens.to(self.device)).to(next(self.llm.parameters()).dtype)

    @torch.no_grad()
    def generate(self, source_input, prompt="", max_new=50, temperature=0.8,
                 top_p=0.9, repetition_penalty=1.0):
        self.bridge.eval()
        self.llm.eval()
        prefix = self._extract_prefix(source_input)
        if prompt:
            prompt_ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
            prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)
            curr = torch.cat([prefix, prompt_embeds], dim=1)
        else:
            curr = prefix
        gen_ids, past, eos_id = [], None, self.tok.eos_token_id
        for _ in range(max_new):
            out = self.llm(inputs_embeds=curr if past is None else curr[:, -1:, :],
                           past_key_values=past, use_cache=True)
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
        text = self.tok.decode(gen_ids) if gen_ids else "(empty)"
        for t in ("<|user|>", "<|assistant|>", "</s>", "<s>", "<pad>"):
            text = text.replace(t, "")
        return text.strip()

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.bridge.state_dict(), os.path.join(path, "bridge.pt"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({"source_encoder": self.source_encoder.name, "llm_name": self.llm_name,
                       "llm_dim": self.llm_dim, "bridge_type": self.bridge_type}, f, indent=2)

    def load(self, path):
        if os.path.exists(os.path.join(path, "config.json")):
            with open(os.path.join(path, "config.json")) as f:
                cfg = json.load(f)
            if cfg.get("bridge_type", self.bridge_type) != self.bridge_type:
                self.bridge = get_bridge(cfg["bridge_type"], self.source_encoder.dim, self.llm_dim)
                self.bridge_type = cfg["bridge_type"]
        self.bridge.load_state_dict(torch.load(os.path.join(path, "bridge.pt"), map_location=self.device))
        self.bridge.to(self.device).eval()

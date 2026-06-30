"""Run benchmarks: bridge vs fusion across all modalities."""
import os, time, math, random, json, warnings
warnings.filterwarnings("ignore")
import torch

from fussion.fusion import FusionLLM, train_fusion
from fussion.merger import CrossModalMerger
from fussion.encoders import CLIPEncoder, get_encoder
from fussion.datasets import (
    make_shape_dataset,
    make_video_dataset,
    make_code_dataset,
    make_text_dataset,
)

DEVICE = "cpu"
random.seed(42)
torch.manual_seed(42)
OUT = os.path.dirname(os.path.abspath(__file__))
results = []


def compute_fusion_ppl(fusion, val_src, val_tgt, encoder, device=DEVICE):
    """Compute perplexity for a trained FusionLLM."""
    val_vis_list = []
    for x in val_src:
        v = encoder([x])
        B, L, D = v.shape
        if L > 50:
            v = v[:, :50, :]
        elif L < 50:
            v = torch.cat([v, torch.zeros(B, 50 - L, D)], dim=1)
        val_vis_list.append(v)
    val_vis = torch.cat(val_vis_list, dim=0)

    fusion.eval()
    total_loss = 0.0
    for i in range(len(val_src)):
        vis = val_vis[i : i + 1].to(device)
        enc = fusion.tok(
            [val_tgt[i]],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        ids = enc.input_ids.to(device)
        mask = enc.attention_mask.to(device)
        labels = ids.clone()
        labels[mask == 0] = -100
        out = fusion(
            ids[:, :-1],
            vis,
            labels=labels[:, 1:],
            attention_mask=mask[:, :-1],
        )
        total_loss += out["loss"].item()

    return math.exp(total_loss / max(len(val_src), 1))


def compute_bridge_ppl(merger, val_src, val_tgt):
    """Compute perplexity for a trained bridge."""
    return merger.evaluate(val_src, val_tgt)


# ── Image: Bridge vs Fusion ──
print("=" * 60)
print("  IMAGE: Bridge vs Fusion")
print("=" * 60)
(train_src, train_tgt), (val_src, val_tgt) = make_shape_dataset(n=80)
encoder = CLIPEncoder(device=DEVICE)

merger = CrossModalMerger(
    source_encoder="clip", target_llm="gpt2", bridge_type="mlp", device=DEVICE
)
t0 = time.time()
merger.train_bridge(
    train_src,
    train_tgt,
    val_source=val_src,
    val_targets=val_tgt,
    steps=20,
    batch_size=8,
    eval_every=10,
)
bridge_ppl = compute_bridge_ppl(merger, val_src, val_tgt)
bridge_t = time.time() - t0
bridge_p = sum(p.numel() for p in merger.bridge.parameters())
print(f"  -> Bridge: PPL={bridge_ppl:.2f}  params={bridge_p:,}  time={bridge_t:.0f}s")
results.append(
    {
        "modality": "image",
        "approach": "bridge",
        "ppl": round(bridge_ppl, 2),
        "params": bridge_p,
        "time": round(bridge_t),
    }
)

fusion = FusionLLM(
    "gpt2", encoder_dim=512, every_k_layers=4, device=DEVICE, verbose=True
)
t0 = time.time()
train_fusion(fusion, encoder, train_src, train_tgt, val_src, val_tgt, steps=20, batch_size=4)
fusion_t = time.time() - t0
fusion_p = sum(p.numel() for p in fusion.get_trainable_params())
fusion_ppl = compute_fusion_ppl(fusion, val_src, val_tgt, encoder)
print(f"  -> Fusion: PPL={fusion_ppl:.2f}  params={fusion_p:,}  time={fusion_t:.0f}s")
results.append(
    {
        "modality": "image",
        "approach": "fusion",
        "ppl": round(fusion_ppl, 2),
        "params": fusion_p,
        "time": round(fusion_t),
    }
)

# ── Video, Code, Text: Fusion only ──
for mod_name, make_data, enc_fn, enc_dim in [
    ("video", make_video_dataset, lambda: get_encoder("video"), 512),
    ("code", make_code_dataset, lambda: get_encoder("codebert"), 768),
    ("text", make_text_dataset, lambda: get_encoder("text_llm"), 768),
]:
    print(f"\n{'=' * 60}")
    print(f"  {mod_name.upper()}: Fusion")
    print(f"{'=' * 60}")
    (train_src, train_tgt), (val_src, val_tgt) = make_data(n=40)
    encoder = enc_fn()
    fusion = FusionLLM(
        "gpt2", encoder_dim=enc_dim, every_k_layers=4, device=DEVICE, verbose=False
    )
    t0 = time.time()
    train_fusion(fusion, encoder, train_src, train_tgt, val_src, val_tgt, steps=15, batch_size=4)
    t = time.time() - t0
    p = sum(p.numel() for p in fusion.get_trainable_params())
    ppl = compute_fusion_ppl(fusion, val_src, val_tgt, encoder)
    print(f"  -> Fusion: PPL={ppl:.2f}  params={p:,}  time={t:.0f}s")
    results.append(
        {
            "modality": mod_name,
            "approach": "fusion",
            "ppl": round(ppl, 2),
            "params": p,
            "time": round(t),
        }
    )

# ── Summary ──
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
header = f"{'Modality':<12} {'Approach':<12} {'PPL':>8} {'Params':>12} {'Time':>8}"
print(header)
print("-" * len(header))
for r in results:
    print(
        f"{r['modality']:<12} {r['approach']:<12} {r['ppl']:>8.2f} {r['params']:>12,} {r['time']:>8.0f}s"
    )

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {os.path.join(OUT, 'results.json')}")

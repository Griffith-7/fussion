"""CLI: python -m fussion [command]."""
import argparse
import torch
from .merger import CrossModalMerger
from .fusion import FusionLLM, train_fusion
from .encoders import get_encoder, list_modalities


def main():
    parser = argparse.ArgumentParser(prog="fussion", description="Cross-modal fusion toolkit")
    sub = parser.add_subparsers(dest="command")

    # list-modalities
    sub.add_parser("list-modalities", help="List available source encoders")

    # bridge
    bridge_p = sub.add_parser("bridge", help="Train an external bridge")
    bridge_p.add_argument("--encoder", default="clip", choices=list_modalities())
    bridge_p.add_argument("--llm", default="gpt2")
    bridge_p.add_argument("--type", default="mlp", choices=["linear", "mlp", "transformer"])
    bridge_p.add_argument("--steps", type=int, default=20)
    bridge_p.add_argument("--batch-size", type=int, default=8)

    # fusion
    fusion_p = sub.add_parser("fusion", help="Train an internal fusion model")
    fusion_p.add_argument("--encoder", default="clip", choices=list_modalities())
    fusion_p.add_argument("--llm", default="gpt2")
    fusion_p.add_argument("--steps", type=int, default=20)
    fusion_p.add_argument("--every-k", type=int, default=4)
    fusion_p.add_argument("--batch-size", type=int, default=4)

    args = parser.parse_args()

    if args.command == "list-modalities":
        print("Available encoders:", list_modalities())
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from .datasets import make_shape_dataset as dataset_fn
    (train_src, train_tgt), (val_src, val_tgt) = dataset_fn(n=80)

    if args.command == "bridge":
        m = CrossModalMerger(args.encoder, args.llm, bridge_type=args.type, device=device)
        m.train_bridge(train_src, train_tgt, val_src, val_tgt,
                       steps=args.steps, batch_size=args.batch_size)
    elif args.command == "fusion":
        enc = get_encoder(args.encoder, device=device)
        enc_dim = enc.dim
        f = FusionLLM(args.llm, encoder_dim=enc_dim, every_k_layers=args.every_k, device=device)
        train_fusion(f, enc, train_src, train_tgt, val_src, val_tgt,
                     steps=args.steps, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

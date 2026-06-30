"""CLI: python -m fussion [command]."""

import argparse
import logging
import sys

import torch

from .datasets import make_shape_dataset
from .encoders import get_encoder, list_modalities
from .exceptions import BridgeNotFoundError, EncoderNotFoundError
from .fusion import FusionLLM, train_fusion
from .merger import CrossModalMerger


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(message)s",
        level=level,
        stream=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="fussion", description="Cross-modal fusion toolkit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list-modalities", help="List available source encoders")

    bridge_p = sub.add_parser("bridge", help="Train an external bridge")
    bridge_p.add_argument("--encoder", default="clip", choices=list_modalities())
    bridge_p.add_argument("--llm", default="gpt2")
    bridge_p.add_argument("--type", default="mlp", choices=["linear", "mlp", "transformer"])
    bridge_p.add_argument("--steps", type=int, default=20)
    bridge_p.add_argument("--batch-size", type=int, default=8)

    fusion_p = sub.add_parser("fusion", help="Train an internal fusion model")
    fusion_p.add_argument("--encoder", default="clip", choices=list_modalities())
    fusion_p.add_argument("--llm", default="gpt2")
    fusion_p.add_argument("--steps", type=int, default=20)
    fusion_p.add_argument("--every-k", type=int, default=4)
    fusion_p.add_argument("--batch-size", type=int, default=4)

    args = parser.parse_args()
    _setup_logging(getattr(args, "verbose", False))

    if args.command == "list-modalities":
        print("Available encoders:", list_modalities())
        return

    if args.command is None:
        parser.print_help()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    (train_src, train_tgt), (val_src, val_tgt) = make_shape_dataset(n=80)

    try:
        if args.command == "bridge":
            m = CrossModalMerger(args.encoder, args.llm, bridge_type=args.type, device=device)
            m.train_bridge(
                train_src, train_tgt, val_src, val_tgt,
                steps=args.steps, batch_size=args.batch_size,
            )
        elif args.command == "fusion":
            enc = get_encoder(args.encoder, device=device)
            f = FusionLLM(args.llm, encoder_dim=enc.dim, every_k_layers=args.every_k, device=device)
            train_fusion(f, enc, train_src, train_tgt, val_src, val_tgt,
                         steps=args.steps, batch_size=args.batch_size)
    except (EncoderNotFoundError, BridgeNotFoundError) as e:
        logging.error("Configuration error: %s", e)
        sys.exit(1)
    except Exception as e:
        logging.error("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

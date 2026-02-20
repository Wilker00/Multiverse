"""
tools/vae_demo.py

Encode/decode a JSON vector using the VAE stub.
"""

from __future__ import annotations

import argparse
import json

import os
import sys

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from memory.vae_stub import VAEConfig, VAEStub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector", type=str, required=True, help="JSON list of floats")
    ap.add_argument("--latent_dim", type=int, default=4)
    args = ap.parse_args()

    vec = json.loads(args.vector)
    cfg = VAEConfig(input_dim=len(vec), latent_dim=args.latent_dim)
    vae = VAEStub(cfg)
    z = vae.encode(vec)
    x = vae.decode(z)
    print(f"latent: {z}")
    print(f"recon: {x}")


if __name__ == "__main__":
    main()





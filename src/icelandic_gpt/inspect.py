"""Measure actual parameter/gradient/Adam-state bytes by transformer block."""
import argparse
import json
from pathlib import Path

import torch

from .model import GPT
from .training import autocast_context, read_config, setup_runtime


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def memory_report(model, optimizer):
    def summarize(parameters):
        parameters = list(parameters)
        return {
            "parameters": sum(p.numel() for p in parameters),
            "parameter_bytes": sum(tensor_bytes(p) for p in parameters),
            "gradient_bytes": sum(tensor_bytes(p.grad) for p in parameters if p.grad is not None),
            "adam_moment_bytes": sum(
                tensor_bytes(state[key])
                for p in parameters for state in [optimizer.state.get(p, {})]
                for key in ("exp_avg", "exp_avg_sq") if key in state
            ),
            "optimizer_scalar_bytes": sum(
                tensor_bytes(state["step"])
                for p in parameters for state in [optimizer.state.get(p, {})]
                if "step" in state
            ),
        }
    return {
        "total": summarize(model.parameters()),
        "blocks": {str(i + 1): summarize(block.parameters()) for i, block in enumerate(model.blocks)},
        "notes": "Measured with allocated gradients after backward and initialized Adam states. "
                 "Includes tied weights once. Excludes activations, logits, allocator and workspaces.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    args = parser.parse_args()
    model_config, config = read_config(args.config)
    device = setup_runtime(args.device, config.seed, config.cpu_threads)
    model = GPT(model_config).to(device)
    optimizer = model.configure_optimizer()
    x = torch.randint(model_config.vocab_size, (1, min(4, model_config.block_size)), device=device)
    with autocast_context(device, config.precision):
        _, loss = model(x, x)
    loss.backward()
    optimizer.step()  # Adam states are initialized lazily.
    report = memory_report(model, optimizer)
    print(json.dumps(report, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

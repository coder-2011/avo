# AVO Experiments

This is the live lab notebook for building the Ampere AVO implementation.
The architecture reference remains `arch.md`; this file records concrete
implementation choices, checks, research, and open questions.

## 2026-05-07 - Checkpoint 0: implementation scope

Success criteria for the first meaningful checkpoint:

- Create the AVO implementation in a separate git repository from this paper repo.
- Target RTX A6000 / sm_86 explicitly; FA4 is excluded.
- Represent the benchmark suite from `arch.md`: BF16 forward attention, head dim 128,
  sequence lengths 4096, 8192, 16384, 32768, total tokens 32768, causal and non-causal.
- Provide isolated child-process execution for scoring so CUDA crashes do not kill the
  orchestrator.
- Provide a compilation gate that invokes `nvcc` with the explicit
  `-gencode=arch=compute_86,code=sm_86` target.
- Provide a correctness reference based on PyTorch `scaled_dot_product_attention`.
- Provide TFLOPS calculation using standard FlashAttention-style forward FLOP counts,
  with causal cases counted at half the dense attention work.
- Provide a FlashAttention-2 baseline hook when `flash_attn` is installed, while keeping
  tests runnable without it.
- Provide a minimal Anthropic variation-agent wrapper that uses `ANTHROPIC_API_KEY` and
  asks for structured JSON decisions.
- Provide deterministic transcript compaction and local recovery for JSON embedded in
  otherwise malformed agent text, while still rejecting invalid decision payloads.
- Add tests for configuration, FLOP math, subprocess isolation, lineage gating, and agent
  output validation.

Local environment findings:

- GPU: NVIDIA RTX A6000, compute capability 8.6, 49140 MiB.
- CUDA compiler: `/usr/local/cuda-12.9/bin/nvcc`, release 12.9.
- PyTorch: 2.11.0+cu129 with CUDA available on the A6000.
- Global Python packages missing: `flash_attn`, `anthropic`.
- `.env.local` contains `ANTHROPIC_API_KEY`; code should read the key from the environment
  and optionally load `.env.local` without logging secrets.
- I could not find a local "pod High agent implementation" source tree under `/home/ubuntu`.
  The only nearby High-related artifact is a highlevel app/plugin manifest, not implementation
  code. I will use the local `pi-mono-agent` package for pattern reading only: evented loop,
  validated tool calls, error tool results, and context transformation/compaction hooks.

Online research notes:

- NVIDIA Ampere tuning guide: compute capability 8.6 should be compiled explicitly for 8.6;
  sm_86 has 100 KB shared memory per SM, 99 KB max shared memory per block, cp.async support,
  third-generation tensor cores, and no Blackwell TMA/WGMMA path.
  Source: https://docs.nvidia.com/cuda/archive/11.6.2/ampere-tuning-guide/index.html
- FlashAttention-2 official repository supports Ampere/Ada/Hopper, BF16 on Ampere+, and head
  dims up to 256 for forward. This supports using FA2 as the seed and baseline on A6000.
  Source: https://github.com/Dao-AILab/flash-attention
- PyTorch `torch.nn.functional.scaled_dot_product_attention` is the correctness reference for
  SDPA semantics. Source: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- Anthropic structured outputs and strict tool use are the right reliability direction for
  machine-parseable agent decisions. Source: https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs
- Anthropic context editing and SDK compaction are relevant for long runs with heavy tool output.
  Source: https://docs.anthropic.com/en/docs/build-with-claude/context-editing

Tradeoffs and uncertainty:

- This checkpoint will not mutate or improve a CUDA attention kernel yet. It builds the minimum
  scoring and lineage substrate needed before autonomous optimization can be trusted.
- `flash_attn` is not globally installed; installing/building it on CUDA 12.9 may be a separate
  checkpoint because it can be slow and environment-sensitive.
- The initial Anthropic wrapper uses JSON schema output when the installed SDK supports it and
  validates the returned JSON locally either way. Tool execution can be added after the score
  gates are stable.

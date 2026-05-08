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
- FA2 package probe: `uv run --extra baseline ... import flash_attn` initially failed because
  `flash-attn==2.8.3` imports `torch` during build metadata generation without declaring it as
  a build dependency. I encoded this in `pyproject.toml` with `tool.uv.extra-build-dependencies`
  for `flash-attn`.

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
- FlashAttention upstream source `setup.py` uses `FLASH_ATTN_CUDA_ARCHS` to control build-gencode targets and defaults to
  `80;90;100;110;120`; `add_cuda_gencodes` is called from that variable. Source:
  https://github.com/Dao-AILab/flash-attention/blob/main/setup.py
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

## 2026-05-07 - Checkpoint 1: FA2 baseline and lineage seeding

- Added `seed-baseline` CLI command in `avo-ampere` to run `worker-score` with
  FlashAttention-2, persist `scores/baseline.json`, and refresh `scores/latest.json`.
- Added lineage helper `seed_baseline` for recording and committing a seed baseline
  before candidate evolution.
- Set the baseline scorer's environment path to `FLASH_ATTN_CUDA_ARCHS=80` to
  keep FlashAttention builds targeted to Ampere-family kernels for this workflow.
- Added regression test for seeded baseline payload persistence in `tests/test_lineage.py`.

## 2026-05-07 - Checkpoint 1.1: baseline-path verification and guard

- Verified scaffold-level FA2 path through a direct CLI smoke run:
  `uv run --extra baseline python -m avo score --backend flash-attn --seq-lens 4096 --causal both --repeats 1 --warmup 1`.
  The command attempted to build `flash-attn`, confirming dependency installation
  is environment-sensitive and may require explicit install time.
- Added `tests/test_cli.py` to assert `_baseline_build_env()` enforces
  `FLASH_ATTN_CUDA_ARCHS=80` for baseline scoring.
- Exa research pass confirmed the FlashAttention-2 support matrix still lists
  Ampere/Ada/Hopper GPUs, BF16 requiring Ampere or newer, and head dimensions up
  to 256. This supports the A6000/sm_86, BF16, head-dim-128 baseline choice.
  Source: https://github.com/Dao-AILab/flash-attention

## 2026-05-08 - Checkpoint 1.2: stricter agent decision boundary

Success criteria for this checkpoint:

- Keep the agent path in the Anthropic ecosystem and continue reading `ANTHROPIC_API_KEY`
  from the environment.
- Prefer a strict, schema-constrained Anthropic tool call for variation decisions.
- Preserve the previous JSON text parser as a fallback for SDK/model combinations that do
  not accept the strict tool-use request.
- Locally validate every decision payload before the orchestrator trusts it.
- Add tests that cover strict schema construction, tool-use parsing, text fallback, and
  rejection when the model calls the wrong tool.

Implementation:

- Added `record_variation_decision` as a strict Anthropic client tool in
  `/home/ubuntu/avo-ampere/avo/agent.py`.
- Updated `request_variation_decision` to request that exact tool via
  `tool_choice={"type": "tool", "name": "record_variation_decision"}` and
  `disable_parallel_tool_use=True`.
- Added `parse_decision_response` so tool-use blocks are preferred, but plain text JSON
  remains a validated fallback.
- Left actual file editing/tool execution out of this checkpoint. The agent still produces
  one structured variation proposal; command execution and mutation loops should be gated
  separately after the decision boundary is stable.

Online research notes:

- Exa research found current Anthropic docs recommending structured outputs for guaranteed
  schema-conforming JSON and strict tool use for guaranteed tool-input schema conformance.
  Source: https://docs.claude.com/en/docs/build-with-claude/structured-outputs
- Exa research found the current strict tool-use API shape: add `strict: True` beside
  `name`, `description`, and `input_schema`; the generated `tool_use.input` follows that
  schema. Source: https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use
- Exa research confirmed `tool_choice` can force a specific tool and disable parallel tool
  use for exactly one tool call. Source:
  https://console.anthropic.com/docs/en/api/python/messages/create

Verification:

- `uv run --extra dev ruff check .` in `/home/ubuntu/avo-ampere`: passed.
- `uv run --extra dev pytest` in `/home/ubuntu/avo-ampere`: 25 passed.

Tradeoffs and uncertainty:

- Strict tool use strengthens the decision format, but it does not yet make the AVO loop
  autonomous. The next missing piece is an allowlisted variation-step executor that can take
  a validated decision, run a bounded command, record the attempt, and commit only through
  the existing correctness/throughput gate.

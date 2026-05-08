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

## 2026-05-08 - Checkpoint 1.3: bounded decision command executor

Success criteria for this checkpoint:

- Add a minimal execution bridge from a validated variation decision to one bounded local
  command.
- Do not use a shell and do not allow arbitrary commands from the model.
- Keep commit/lineage acceptance outside the model-controlled command path.
- Record the attempted decision and command result as JSON for later diagnosis.
- Verify command parsing, rejection behavior, execution, and attempt persistence with tests.

Implementation:

- Added `/home/ubuntu/avo-ampere/avo/evolve.py` with:
  - `command_from_decision`, which accepts only commands beginning with `avo`.
  - A default subcommand allowlist of `env`, `compile`, and `score`.
  - Rejection of shell control tokens such as `&&`, `|`, and redirection.
  - `run_decision_command`, which rewrites the command to `python -m avo ...` and runs it
    with `subprocess.run(..., shell=False, timeout=...)`.
  - `write_attempt`, which persists a structured attempt JSON artifact.
- Added `avo run-decision <decision.json> --attempt-json <path>` as a CLI entry point.
- Updated the README with the `agent-plan` -> persisted JSON -> `run-decision` workflow.
- The unit tests use a test-only allowlist for `worker-sleep` so they do not import CUDA or
  require GPU work. The production default allowlist remains `env`, `compile`, and `score`.

Online/local research notes:

- Anthropic Agent SDK docs emphasize constraining tools with allowed tool lists and using
  hooks to validate, log, block, or transform agent behavior. The local implementation is
  not using the SDK yet, but this checkpoint mirrors that reliability pattern with a small
  allowlist and a JSON attempt log. Source:
  https://console.anthropic.com/docs/en/agent-sdk/structured-outputs
- The local `pi-mono-agent` package uses validated tool calls, pre/post tool hooks, and
  ordered tool-result persistence. I used it only as a pattern reference; no code was copied.

Verification:

- `uv run --extra dev ruff check .` in `/home/ubuntu/avo-ampere`: passed.
- `uv run --extra dev pytest` in `/home/ubuntu/avo-ampere`: 30 passed.

Tradeoffs and uncertainty:

- `run-decision` is intentionally not a full mutation loop. It can execute bounded AVO
  subcommands and record attempts, but it cannot edit files, cannot run arbitrary shell,
  and cannot directly bypass the lineage gate.
- The next missing piece is candidate-kernel support: a concrete seed/candidate CUDA
  extension path that can be compiled, imported in an isolated scorer, compared to PyTorch
  SDPA, and then committed only via `commit_score`.

## 2026-05-08 - Checkpoint 1.4: real A6000 runtime smoke verification

Success criteria for this checkpoint:

- Verify the local runtime still sees the intended A6000/sm_86 target.
- Verify the compile gate invokes NVCC with `compute_86,sm_86`.
- Verify the isolated scorer can run a real CUDA Torch SDPA benchmark and return structured
  JSON without crashing the orchestrator.

Commands and results:

- `uv run --extra cuda python -m avo env` in `/home/ubuntu/avo-ampere`: passed.
  - GPU: NVIDIA RTX A6000.
  - Compute capability: 8.6.
  - Target gencode: `-gencode=arch=compute_86,code=sm_86`.
  - PyTorch: `2.11.0+cu130`, CUDA available.
- `uv run python -m avo compile --source kernels/smoke.cu --out-dir /tmp/avo-build-smoke`:
  passed. The command used `/usr/local/cuda-12.9/bin/nvcc` and the explicit
  `-gencode=arch=compute_86,code=sm_86` flag.
- `uv run --extra cuda python -m avo score --backend torch-sdpa --seq-lens 4096 --causal both --repeats 1 --warmup 1 --timeout-s 180`:
  passed through the child-process scorer.
  - `all_correct`: true.
  - Non-causal 4096: 9.991 ms, 110.047 TFLOPS.
  - Causal 4096: 5.943 ms, 92.500 TFLOPS.
  - Geomean: 100.893 TFLOPS.

Tradeoffs and uncertainty:

- This is a smoke run, not a stable benchmark. `repeats=1` is only enough to verify the
  orchestration, correctness gate, TFLOPS calculation path, and child-process JSON result.
- Full benchmark runs should use the complete sequence-length suite and more repeats once
  FA2 installation/build time is handled.

## 2026-05-08 - Checkpoint 1.5: candidate scoring interface

Success criteria for this checkpoint:

- Add a candidate backend that can be scored by the same isolated worker path as Torch SDPA
  and FlashAttention-2.
- Load candidate code from an explicit repo-local path rather than hard-coding a single
  implementation.
- Compare candidate output against PyTorch `scaled_dot_product_attention`, use the existing
  tolerance gate, and calculate TFLOPS with the same math as the other backends.
- Keep the initial seed candidate deliberately boring so the scorer contract is verified
  before attempting a custom CUDA attention kernel.
- Verify with unit tests and one real A6000 candidate smoke run.

Implementation:

- Added `candidate` as a scorer backend in `/home/ubuntu/avo-ampere/avo/benchmark.py`.
- Added `--candidate <path>` to `score` and `worker-score`.
- Candidate modules must define:
  `attention(q, k, v, causal: bool)`.
- The scorer passes tensors in PyTorch SDPA layout:
  `(batch, heads, seq, head_dim)`.
- Added `/home/ubuntu/avo-ampere/candidates/torch_sdpa_seed.py`, which delegates to PyTorch
  SDPA. This is a correctness seed and scorer-contract fixture, not an optimization.
- Added tests in `/home/ubuntu/avo-ampere/tests/test_candidate_backend.py` for missing
  candidate path, loadable candidate modules, missing `attention` rejection, and structured
  failed scores for candidate load failures.
- Updated the README with the candidate interface and smoke command.

Online research notes:

- Exa research found current PyTorch docs for `torch.utils.cpp_extension` and custom C++/CUDA
  operators. PyTorch supports compiling CUDA sources via `torch.utils.cpp_extension.load` or
  `CUDAExtension`; CUDA files are detected and compiled with NVCC, and additional NVCC flags
  can be supplied with `extra_cuda_cflags`.
  Source: https://docs.pytorch.org/docs/stable/cpp_extension.html
- PyTorch custom operator docs recommend registering real custom ops when calling into custom
  C++/CUDA kernels, while noting that Python bindings around C++/CUDA kernels are possible but
  compose less cleanly with PyTorch subsystems. For this scaffold, hiding any extension build
  behind `attention(...)` is enough until the first real kernel exists.
  Source: https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html

Verification:

- `uv run --extra dev ruff check .` in `/home/ubuntu/avo-ampere`: passed.
- `uv run --extra dev pytest` in `/home/ubuntu/avo-ampere`: 34 passed.
- Real candidate smoke:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/torch_sdpa_seed.py --seq-lens 4096 --causal both --repeats 1 --warmup 1 --timeout-s 180`
  passed through the isolated worker.
  - `all_correct`: true.
  - Candidate path: `candidates/torch_sdpa_seed.py`.
  - Non-causal 4096: 9.682 ms, 113.557 TFLOPS.
  - Causal 4096: 5.866 ms, 93.724 TFLOPS.
  - Geomean: 103.165 TFLOPS.

Tradeoffs and uncertainty:

- This checkpoint enables candidate scoring, but it does not yet add a custom CUDA attention
  kernel. The seed candidate delegates to PyTorch SDPA specifically to prove the scorer
  contract before adding extension build complexity.
- The next candidate-kernel checkpoint should add the smallest CUDA extension-backed candidate
  behind this interface and verify that the extension builds for `sm_86` without weakening
  the correctness gate.

## 2026-05-08 - Checkpoint 1.6: live Anthropic agent-plan smoke

Success criteria for this checkpoint:

- Exercise the real Anthropic `agent-plan` path with `ANTHROPIC_API_KEY` from
  `/home/ubuntu/avo/.env.local`.
- Verify strict tool-use compatibility with the default model.
- Ensure `next_command` is locally bounded to commands the `run-decision` executor can
  actually run.
- Preserve fallbacks for models or SDKs that reject strict tools or structured outputs.

What happened:

- First smoke attempt used the previous default model, `claude-sonnet-4-20250514`.
  Anthropic returned a 400 error: that model does not support strict tools.
- Exa research confirmed structured outputs/strict tool use are generally available on
  Claude Sonnet 4.5 and newer supported models, and Anthropic release notes identify
  `claude-sonnet-4-5-20250929` as the Sonnet 4.5 model ID.
  Source: https://docs.anthropic.com/en/release-notes/api
  Source: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- Updated the default AVO agent model to `claude-sonnet-4-5-20250929`.
- Added API fallback handling:
  - strict tool request -> JSON structured output -> plain JSON text,
  - but only for recognized unsupported-feature errors.
- Added local validation that `next_command` must:
  - start with `avo`;
  - use only `env`, `compile`, or `score`;
  - avoid shell control tokens such as `&&`, pipes, and redirection.
- Added schema descriptions and prompt text telling the model not to emit arbitrary shell,
  `cat`, `head`, `git`, or destructive commands.

Verification:

- `uv run --extra dev ruff check .` in `/home/ubuntu/avo-ampere`: passed.
- `uv run --extra dev pytest` in `/home/ubuntu/avo-ampere`: 40 passed.
- Live command:
  `uv run --extra agent python -m avo agent-plan --lineage /tmp/avo-agent-smoke-lineage --knowledge knowledge/ampere.md --env-file /home/ubuntu/avo/.env.local`
  passed.
- The live response returned a validated bounded command:
  `next_command: "avo env"`.

Tradeoffs and uncertainty:

- The agent can still propose file-inspection paths that are not present in `avo-ampere`
  when the lineage is empty and the prompt points it toward upstream FA2. That is acceptable
  for this checkpoint because the executable command is bounded; future prompts should include
  a more precise repo file inventory once the first candidate kernel files exist.

## 2026-05-08 - Checkpoint 1.7: CUDA extension-backed candidate smoke

Success criteria for this checkpoint:

- Add the smallest candidate module that actually builds and invokes custom CUDA code behind
  the existing `attention(q, k, v, causal)` scorer interface.
- Ensure the extension build targets A6000/sm_86.
- Keep the correctness gate unchanged: output must still compare against PyTorch SDPA.
- Do not claim this is an optimized attention kernel.

Implementation:

- Added `/home/ubuntu/avo-ampere/candidates/cuda_identity_seed.py`.
- Added CUDA extension sources under `/home/ubuntu/avo-ampere/candidates/cuda_identity/`:
  - `identity.cpp` exposes `identity(tensor)` via pybind.
  - `identity_kernel.cu` launches a simple CUDA copy kernel supporting BF16/FP16/FP32/FP64
    dispatch through PyTorch's scalar dispatch macros.
- `cuda_identity_seed.py` computes attention with PyTorch SDPA, then passes the output through
  the custom CUDA identity kernel. This proves extension build/load/invocation without moving
  attention math into the custom kernel yet.
- Added `ninja` to the `cuda` extra because `torch.utils.cpp_extension.load` requires it.
- The module sets `TORCH_CUDA_ARCH_LIST=8.6` and `MAX_JOBS=2` by default for controlled local
  builds.

Online research notes:

- Exa research confirmed PyTorch extension builds can be targeted explicitly with
  `TORCH_CUDA_ARCH_LIST`, including `8.6`, and that specifying exact compute capabilities is
  preferable when the target GPU is known.
  Source: https://docs.pytorch.org/docs/stable/cpp_extension.html

Verification and debugging:

- First extension smoke failed as a structured candidate score because `ninja` was missing.
  This verified that extension failures stay inside the scorer payload instead of crashing the
  orchestrator.
- After adding `ninja`, the next build reached NVCC but failed on a C++ declaration mismatch
  in `identity.cpp`; fixed by changing the forward declaration to `void identity_cuda(...)`.
- Clean rebuild command:
  `rm -rf /home/ubuntu/.cache/torch_extensions/py312_cu130/avo_cuda_identity_seed && AVO_VERBOSE_EXT_BUILD=1 uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_identity_seed.py --seq-lens 4096 --causal false --repeats 1 --warmup 1 --timeout-s 300`
  passed.
- The clean build output showed:
  `/usr/local/cuda-12.9/bin/nvcc ... -gencode=arch=compute_86,code=sm_86 ... identity_kernel.cu`.
- Result:
  - `all_correct`: true.
  - Non-causal 4096: 10.068 ms, 109.204 TFLOPS.
  - Geomean: 109.204 TFLOPS.
- `uv run --extra dev ruff check .` in `/home/ubuntu/avo-ampere`: passed.
- `uv run --extra dev pytest` in `/home/ubuntu/avo-ampere`: 40 passed.

Tradeoffs and uncertainty:

- This is not an evolved attention kernel. It is a CUDA extension-backed correctness and build
  smoke. The actual AVO kernel work still needs to replace the SDPA call with custom attention
  math while keeping the same scorer interface and lineage gate.
- Build output is captured in the isolated worker's stdout tail. For long compiler failures,
  the 4000-character tail may omit early lines; this was enough for the smoke but may need a
  larger artifact log during real kernel development.

## 2026-05-08 - Checkpoint 1.8: one-step AVO orchestration

Success criteria for this checkpoint:

- Add a minimal command that performs one AVO variation step:
  request a validated Anthropic decision, run one bounded command, persist the step, and
  apply the lineage gate only if a score payload was produced.
- Keep the score gate authoritative; the agent must not be able to commit directly.
- Preserve non-score attempts as diagnostic artifacts without mutating lineage.
- Verify the step logic with tests and run a live Anthropic smoke.

Implementation:

- Added `EvolutionStep` in `/home/ubuntu/avo-ampere/avo/evolve.py`.
- `run_decision_command` now extracts a score payload from either:
  - the outer `avo score` isolated-result JSON, or
  - a direct `AVO_RESULT_JSON=...` worker line.
- Added `finalize_attempt(lineage, attempt)`, which calls `commit_score` only when the
  attempt contains a score payload.
- Added `write_step` for persisted JSON step artifacts.
- Added `avo evolve-once --lineage <path> --knowledge <path> --step-json <path>`.
- Updated README with the `evolve-once` workflow.

Online research notes:

- Exa research on the Claude Agent SDK loop emphasized the same control split used here:
  the model chooses a tool/action, but the application controls allowed tools, permissions,
  hooks, and logging. Source: https://code.claude.com/docs/en/agent-sdk/agent-loop
- Exa research on hooks/permissions highlighted pre-tool blocking and post-tool audit logging
  as standard reliability patterns. This checkpoint implements the same idea locally with the
  existing bounded command executor and JSON step artifacts rather than adopting the SDK yet.
  Source: https://code.claude.com/docs/en/agent-sdk/hooks

Verification:

- `uv run --extra dev ruff check .` in `/home/ubuntu/avo-ampere`: passed.
- `uv run --extra dev pytest` in `/home/ubuntu/avo-ampere`: 45 passed.
- Live command:
  `uv run --extra agent python -m avo evolve-once --lineage /tmp/avo-evolve-smoke-lineage --knowledge knowledge/ampere.md --env-file /home/ubuntu/avo/.env.local --step-json /tmp/avo-evolve-smoke-step.json --timeout-s 180`
  passed.
- Live result:
  - Anthropic returned bounded `next_command: "avo env"`.
  - The executor ran `python -m avo env` successfully.
  - The command output confirmed RTX A6000, compute capability 8.6, and
    `-gencode=arch=compute_86,code=sm_86`.
  - `score_payload`: null.
  - `gate_decision`: null, so lineage did not receive an ungated/non-score commit.

Tradeoffs and uncertainty:

- This is one-step orchestration, not a full autonomous editing loop. It can safely run bounded
  environment/compile/score commands and commit score payloads through the gate, but it cannot
  yet edit candidate source files by itself.
- The agent still tends to propose upstream FA2 files that are not present locally when lineage
  is empty. The next prompt hardening should include a concise repo inventory and current
  candidate paths so the next command moves toward `avo score --backend candidate ...`.

## 2026-05-08 - Checkpoint 1.9: local repo context for agent decisions

Success criteria for this checkpoint:

- Provide the Anthropic planner with concise local repo context instead of only architecture
  notes and lineage state.
- Steer the next bounded command toward local candidate scoring when no accepted lineage score
  exists.
- Verify that `evolve-once` can now produce a score payload and commit only through the
  existing lineage gate.

Implementation:

- Added `build_repo_context(root)` in `/home/ubuntu/avo-ampere/avo/agent.py`.
- The generated context includes:
  - local candidate modules;
  - local CUDA candidate sources;
  - the candidate interface contract;
  - a preferred first local candidate score command for `candidates/cuda_identity_seed.py`;
  - an explicit warning not to propose upstream FA2 `csrc/...` paths unless present locally.
- Added `build_variation_prompt(...)` so prompt construction is testable.
- Wired repo context into both `agent-plan` and `evolve-once`.
- Added `--cwd` to `agent-plan` so callers can choose the repo root used for context.
- Cleaned lineage noise discovered during the live smoke:
  - avoid re-running `git init` when a lineage repo already exists;
  - suppress expected `git show HEAD:scores/latest.json` stderr when no accepted score exists.

Online research notes:

- Exa research on Claude Agent SDK context management emphasized being selective with context
  and tools because tool definitions, prompts, and outputs consume the context window.
  Source: https://code.claude.com/docs/en/agent-sdk/agent-loop
- Exa research also noted project context can be supplied through filesystem/project context
  or directly injected prompts. This checkpoint uses direct prompt context because this scaffold
  still owns its own Anthropic client loop rather than the Agent SDK.
  Source: https://code.claude.com/docs/en/agent-sdk/claude-code-features

Verification:

- `uv run --extra dev ruff check .` in `/home/ubuntu/avo-ampere`: passed.
- `uv run --extra dev pytest` in `/home/ubuntu/avo-ampere`: 47 passed.
- Live `agent-plan` with empty temp lineage returned local candidate files and:
  `next_command: "avo score --backend candidate --candidate candidates/cuda_identity_seed.py --seq-lens 4096 --causal false --repeats 1 --warmup 1 --timeout-s 300"`.
- Live `evolve-once` with empty temp lineage passed:
  - ran the bounded candidate score command;
  - extracted the score payload;
  - committed through `commit_score`;
  - gate decision accepted the score.
- Live score payload:
  - `all_correct`: true.
  - Candidate path: `candidates/cuda_identity_seed.py`.
  - Non-causal 4096: 10.123 ms, 108.617 TFLOPS.
  - Geomean: 108.617 TFLOPS.

Tradeoffs and uncertainty:

- The prompt is now grounded in local files, but this is still score-first orchestration. It
  does not yet let the agent edit candidate kernels.
- The current CUDA candidate still delegates attention math to PyTorch SDPA and only runs an
  identity CUDA kernel after the fact. The next real implementation step remains moving a small
  attention computation into the candidate kernel while keeping the scorer and lineage gate
  unchanged.

## 2026-05-08 - Checkpoint 2.0: naive CUDA attention math candidate

Success criteria for this checkpoint:

- Put actual scaled dot-product attention math behind the candidate scorer interface.
- Keep the implementation explicitly small and correctness-oriented, not optimized.
- Preserve the default benchmark suite unchanged while allowing tiny smoke shapes for slow
  correctness kernels.
- Verify both causal and non-causal behavior against PyTorch SDPA.
- Verify BF16 works on the A6000, even if the tiny smoke also uses FP32 for stricter numeric
  comparison.

Implementation:

- Added `/home/ubuntu/avo-ampere/candidates/cuda_naive_attention_seed.py`.
- Added CUDA extension sources under
  `/home/ubuntu/avo-ampere/candidates/cuda_naive_attention/`:
  - `attention.cpp`;
  - `attention_kernel.cu`.
- The CUDA kernel is intentionally naive:
  - one CUDA block per `(batch, head, query)` row;
  - one thread computes that entire row;
  - computes `QK^T / sqrt(D)`;
  - applies causal masking by limiting keys to `query + 1`;
  - uses max-subtracted softmax for numerical stability;
  - accumulates in FP32 before casting to the output dtype.
- Added candidate guardrails in Python:
  - `seq_len <= 128`;
  - `head_dim <= 128`.
  This prevents accidental full-suite runs with a deliberately slow seed.
- Added CLI score shape overrides:
  - `--head-dim`;
  - `--num-heads`;
  - `--total-tokens`;
  - `--dtype {bf16,fp16,fp32}`.
- Defaults remain the architecture suite: BF16, head dim 128, 16 heads, total tokens 32768,
  sequence lengths 4096/8192/16384/32768.
- Updated README with the tiny naive-attention smoke command.

Online research notes:

- Exa research found NVIDIA's FlashAttention tuning discussion of numerically stable softmax
  and online softmax. The naive kernel uses the simpler max-subtracted safe softmax form
  before any online/tiled optimization.
  Source: https://developer.nvidia.cn/blog/tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile/
- Exa research also found examples/discussion of attention kernels accumulating dot products,
  softmax statistics, and outputs in FP32 for stability. The seed follows that stability
  pattern while intentionally avoiding tiling and MMA complexity.
  Source: https://arxiv.org/html/2603.01960v1

Verification:

- `uv run --extra dev ruff check .` in `/home/ubuntu/avo-ampere`: passed.
- `uv run --extra dev pytest` in `/home/ubuntu/avo-ampere`: 48 passed.
- Clean FP32 extension smoke:
  `rm -rf /home/ubuntu/.cache/torch_extensions/py312_cu130/avo_cuda_naive_attention_seed && AVO_VERBOSE_EXT_BUILD=1 uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_naive_attention_seed.py --seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16 --dtype fp32 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Build output showed `/usr/local/cuda-12.9/bin/nvcc` with
    `-gencode=arch=compute_86,code=sm_86`.
  - Non-causal FP32: max abs error `7.15e-07`, 1.077 ms.
  - Causal FP32: max abs error `4.77e-07`, 1.000 ms.
- Tiny BF16 smoke:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_naive_attention_seed.py --seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Non-causal BF16: max abs error `0.00390625`, 0.867 ms.
  - Causal BF16: max abs error `0.015625`, 0.645 ms.

Tradeoffs and uncertainty:

- This is the first actual CUDA attention math candidate, but it is not a viable performance
  kernel. It exists to establish correctness, scorer integration, dtype behavior, and sm_86
  extension compilation before moving toward tiled/online-softmax/MMA implementations.
- The scorer now supports tiny smoke shapes for slow kernels. The default architecture
  benchmark remains unchanged, so full-suite results are still comparable to the original
  AVO plan.

## 2026-05-08 - Checkpoint 2.1: steer agent context to the real CUDA attention seed

Success criteria for this checkpoint:

- Remove stale agent guidance that still preferred the identity CUDA smoke candidate.
- Make local repo context prefer the first CUDA candidate that actually computes attention
  math: `candidates/cuda_naive_attention_seed.py`.
- Keep fallback behavior for older/smaller checkouts that only have the identity or Python
  SDPA seeds.
- Expand the Ampere knowledge seed with source-backed constraints that matter for the next
  tiled-online-softmax step.
- Verify the Anthropic planner now returns a bounded score command for the naive attention seed.

Implementation:

- Updated `/home/ubuntu/avo-ampere/avo/agent.py`:
  - added `_preferred_candidate_score_command(...)`;
  - prefer the tiny BF16 `cuda_naive_attention_seed.py` score command when that candidate exists;
  - fall back to `cuda_identity_seed.py`, then `torch_sdpa_seed.py`.
- Updated `/home/ubuntu/avo-ampere/tests/test_agent.py`:
  - assert repo context lists the naive attention seed;
  - assert the preferred command points at the naive attention seed and tiny shape;
  - assert fallback still chooses the identity candidate when the naive seed is absent.
- Updated `/home/ubuntu/avo-ampere/knowledge/ampere.md` with:
  - sm86 occupancy/register/shared-memory facts from NVIDIA's Ampere tuning guide;
  - Ampere `cp.async` framing as the right replacement direction for Blackwell TMA ideas;
  - FlashAttention-2 sm8x block-size heuristic evidence for head dimension 128;
  - a local note that `cuda_identity_seed.py` is now only an extension/build smoke.
- Updated `/home/ubuntu/avo-ampere/README.md` so the missing-work list no longer claims
  there is no CUDA attention candidate at all; the missing work is now a tiled/performance
  candidate beyond the naive seed.

Online research notes:

- Exa research on NVIDIA's Ampere tuning guide confirmed that compute capability 8.6 has
  48 resident warps per SM, 64K 32-bit registers per SM, 16 resident thread blocks per SM,
  100 KB shared memory per SM, and 99 KB maximum shared memory per block. This makes register
  pressure, dynamic shared-memory opt-in, and occupancy explicit constraints for the A6000
  search space.
  Source: https://docs.nvidia.com/cuda/ampere-tuning-guide/
- The same NVIDIA guide describes Ampere async global-to-shared copy support as an overlap
  mechanism that avoids extra copy registers and can bypass L1. This supports orienting the
  next real kernel step around `cp.async`, not Blackwell TMA.
  Source: https://docs.nvidia.com/cuda/ampere-tuning-guide/
- Exa research on FlashAttention-2 v2.8.3 found the sm8x block-size heuristic in
  `flash_attn_interface.py`. For head dimension 128, sm86 is treated separately from sm80
  and chooses smaller N-blocks in some cases. I logged this as search-space evidence rather
  than copying FA2's policy.
  Source: https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/flash_attn/flash_attn_interface.py

Verification:

- `uv run --extra dev ruff check .` in `/home/ubuntu/avo-ampere`: passed.
- `uv run --extra dev pytest tests/test_agent.py` in `/home/ubuntu/avo-ampere`: 17 passed.
- `uv run --extra dev pytest` in `/home/ubuntu/avo-ampere`: 49 passed.
- Live Anthropic planner smoke:
  `uv run --extra agent python -m avo agent-plan --lineage /tmp/avo-agent-context-naive-lineage --knowledge knowledge/ampere.md --cwd /home/ubuntu/avo-ampere --env-file /home/ubuntu/avo/.env.local`
  passed.
  - Returned `files_to_inspect`:
    - `candidates/cuda_naive_attention_seed.py`;
    - `candidates/cuda_naive_attention/attention.cpp`;
    - `candidates/cuda_naive_attention/attention_kernel.cu`.
  - Returned bounded `next_command`:
    `avo score --backend candidate --candidate candidates/cuda_naive_attention_seed.py --seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`.
- Executed the planner-selected command:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_naive_attention_seed.py --seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - `all_correct`: true.
  - Non-causal BF16: max abs error `0.00390625`, 0.734 ms.
  - Causal BF16: max abs error `0.015625`, 0.928 ms.
  - Geomean: `1.4036513857070751e-05` TFLOPS.

Tradeoffs and uncertainty:

- This does not add a new kernel optimization. It fixes a control-plane hazard: the agent was
  about to keep revisiting an obsolete build smoke instead of the first real attention-math
  seed.
- The next implementation step is still kernel work: a tiny tiled online-softmax candidate
  with enough parallelism to replace the one-thread-per-row naive seed, while staying on
  small shapes until BF16/FP32 correctness is boring.

## 2026-05-08 - Checkpoint 2.2: tiny tiled online-softmax seed

Success criteria for this checkpoint:

- Add a new CUDA candidate that moves beyond the one-thread-per-row naive seed without
  overwriting that seed.
- Use an actual tiled online-softmax update: per-tile row max, per-tile denominator,
  output accumulator rescaling, and final normalization.
- Keep the candidate intentionally small: tiny correctness shapes only, no `mma.sync`,
  no `cp.async`, and no FA2-performance claim.
- Make the agent context prefer this new tiled seed for the next bounded candidate score.
- Address a live Anthropic reliability failure observed during verification with a small
  transient-retry wrapper.

Implementation:

- Added `/home/ubuntu/avo-ampere/candidates/cuda_tiled_attention_seed.py`.
- Added CUDA extension sources under
  `/home/ubuntu/avo-ampere/candidates/cuda_tiled_attention/`:
  - `attention.cpp`;
  - `attention_kernel.cu`.
- The CUDA kernel shape is:
  - one CTA per `(batch, head, query)` row;
  - 128 threads per CTA;
  - 32-key score tiles;
  - FP32 score, row max, row sum, and output accumulation;
  - online output rescaling when a later K tile increases the running row max;
  - causal masking by limiting the K loop to `query + 1`.
- Guardrails remain tiny:
  - `seq_len <= 128`;
  - `head_dim <= 128`.
- Updated `/home/ubuntu/avo-ampere/avo/agent.py` so candidate preference is now:
  - `cuda_tiled_attention_seed.py`;
  - `cuda_naive_attention_seed.py`;
  - `cuda_identity_seed.py`;
  - `torch_sdpa_seed.py`.
- Added agent tests for the tiled preference and fallback behavior.
- Added Anthropic request retries for transient API failures:
  - retry status codes: 408, 409, 429, 500, 502, 503, 504, 529;
  - retry connection/timeout-style SDK exceptions by class name;
  - default three attempts with exponential backoff;
  - unit-tested with a fake 500.
- Updated README and `knowledge/ampere.md` to describe the tiled seed as the current local
  candidate.

Online research notes:

- Exa research on minimal FlashAttention examples reinforced the online-softmax structure:
  compute a tile-local max and sum, update running max/sum, and rescale the output accumulator
  when the running max changes.
  Source: https://github.com/tspeterkim/flash-attention-minimal/blob/main/flash.cu
- Exa research on CUDA softmax kernels showed the standard staged pattern for row-wise
  reductions: per-thread local max/sum, block/warp reduction through shared memory, broadcast,
  and normalization. The tiled seed uses the simpler shared-memory block reduction form.
  Source: https://github.com/karpathy/llm.c/blob/master/dev/cuda/softmax_forward.cu
- Exa research on Anthropic API errors confirmed that 500 is an internal API error, 504 is
  timeout, and 529 is overloaded. This supports retrying transient server/rate/timeout
  failures while still surfacing bad requests and schema errors immediately.
  Source: https://docs.anthropic.com/en/api/errors

Verification:

- `uv run --extra dev ruff check .` in `/home/ubuntu/avo-ampere`: passed.
- `uv run --extra dev pytest tests/test_agent.py tests/test_candidate_backend.py`:
  22 passed before adding retry coverage.
- `uv run --extra dev pytest tests/test_agent.py`: 19 passed after adding retry coverage.
- `uv run --extra dev pytest` in `/home/ubuntu/avo-ampere`: 51 passed.
- Clean BF16 tiled extension smoke:
  `rm -rf /home/ubuntu/.cache/torch_extensions/py312_cu130/avo_cuda_tiled_attention_seed && AVO_VERBOSE_EXT_BUILD=1 uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_tiled_attention_seed.py --seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Build output showed `/usr/local/cuda-12.9/bin/nvcc` with
    `-gencode=arch=compute_86,code=sm_86`.
  - Non-causal BF16: max abs error `0.00390625`, 0.910 ms.
  - Causal BF16: max abs error `0.015625`, 0.956 ms.
  - Geomean: `1.2419097337861061e-05` TFLOPS.
- FP32 tiled smoke:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_tiled_attention_seed.py --seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16 --dtype fp32 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Non-causal FP32: max abs error `4.76837158203125e-07`, 0.804 ms.
  - Causal FP32: max abs error `4.76837158203125e-07`, 0.828 ms.
  - Geomean: `1.4203029795027486e-05` TFLOPS.
- Larger BF16 tiled smoke:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_tiled_attention_seed.py --seq-lens 64 --total-tokens 64 --num-heads 1 --head-dim 64 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Non-causal BF16: max abs error `0.00390625`, 0.946 ms.
  - Causal BF16: max abs error `0.0078125`, 0.642 ms.
  - Geomean: `0.000950976739434787` TFLOPS.
- Same 64x64 BF16 shape on the naive seed:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_naive_attention_seed.py --seq-lens 64 --total-tokens 64 --num-heads 1 --head-dim 64 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Non-causal BF16: 4.117 ms.
  - Causal BF16: 4.136 ms.
  - Geomean: `0.00017968202073136562` TFLOPS.
  - This makes the tiled seed about 5.3x higher geomean on this tiny smoke, while still
    remaining many orders of magnitude below a real FA2 target.
- Live Anthropic planner smoke initially failed with an Anthropic 500 internal server error.
  After adding transient retries, rerunning:
  `uv run --extra agent python -m avo agent-plan --lineage /tmp/avo-agent-context-tiled-lineage --knowledge knowledge/ampere.md --cwd /home/ubuntu/avo-ampere --env-file /home/ubuntu/avo/.env.local`
  passed and returned:
  - `files_to_inspect` pointing at `cuda_tiled_attention_seed.py`, `attention_kernel.cu`,
    and `attention.cpp`;
  - bounded `next_command`:
    `avo score --backend candidate --candidate candidates/cuda_tiled_attention_seed.py --seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`.

Tradeoffs and uncertainty:

- This is a structural improvement over the naive seed, not a competitive attention kernel.
  It still computes dot products with scalar loops and does not stage Q/K/V tiles in shared memory.
- The implementation intentionally avoids tensor cores and async copy until the online softmax
  state is stable under small correctness smokes. The next kernel step should introduce either
  multi-row CTA tiling or tensor-core QK/PV fragments, but not both at once.

## 2026-05-08 - Checkpoint 2.3: warp-row multi-row attention seed

Success criteria for this checkpoint:

- Add a separate CUDA candidate that introduces multi-row CTA structure without mutating the
  working one-CTA-per-row tiled seed.
- Use one warp per query row and multiple rows per CTA, while preserving the existing online
  softmax math and tiny-shape guardrails.
- Use warp-level reductions for row max and row sum, not block-wide shared-memory reductions.
- Keep the candidate intentionally scalar: no `mma.sync`, no `cp.async`, no shared K/V staging.
- Verify BF16 and FP32 correctness against PyTorch SDPA.
- Make the Anthropic planner prefer this new seed for the next bounded score command.

Implementation:

- Added `/home/ubuntu/avo-ampere/candidates/cuda_warp_rows_attention_seed.py`.
- Added CUDA extension sources under
  `/home/ubuntu/avo-ampere/candidates/cuda_warp_rows_attention/`:
  - `attention.cpp`;
  - `attention_kernel.cu`.
- The CUDA kernel shape is:
  - 4 query rows per CTA;
  - 1 warp per query row;
  - 32-key tiles;
  - warp-shuffle max and sum reductions;
  - per-lane output accumulators for up to head dimension 128;
  - FP32 score, row max, row sum, and output accumulation;
  - online output rescaling when a later K tile increases the running row max;
  - causal masking by limiting each row's K loop to `query + 1`.
- Updated `/home/ubuntu/avo-ampere/avo/agent.py` so candidate preference is now:
  - `cuda_warp_rows_attention_seed.py`;
  - `cuda_tiled_attention_seed.py`;
  - `cuda_naive_attention_seed.py`;
  - `cuda_identity_seed.py`;
  - `torch_sdpa_seed.py`.
- Added agent tests for warp-row preference and fallback to the tiled seed.
- Updated README and `knowledge/ampere.md` to describe the warp-row seed as the current
  local candidate.

Online research notes:

- Exa research on CUDA softmax examples showed warp-level max/sum reductions using shuffle
  intrinsics before shared-memory inter-warp reduction. This checkpoint uses only the warp
  portion because each warp owns one row.
  Source: https://github.com/karpathy/llm.c/blob/master/dev/cuda/attention_forward.cu
- Exa research on online-softmax examples showed a fused max/denominator reduction pattern
  using `__shfl_xor_sync`. I kept the implementation simpler by separately reducing tile max
  and tile sum, then applying the same online rescaling formula used in prior checkpoints.
  Source: https://github.com/DefTruth/CUDA-Learn-Notes/blob/main/kernels/softmax/softmax.cu
- Exa research on FlashAttention-oriented kernel writeups reinforced that in-register or
  shuffle-based softmax state is a real optimization direction before, or alongside, tensor-core
  fragment work. This checkpoint uses that direction without copying the PTX/MMA design.
  Source: https://github.com/Tugbars/Flash-Attention-PTX-CUDA

Verification:

- `uv run --extra dev ruff check .` in `/home/ubuntu/avo-ampere`: passed.
- `uv run --extra dev pytest tests/test_agent.py tests/test_candidate_backend.py`:
  24 passed.
- `uv run --extra dev pytest` in `/home/ubuntu/avo-ampere`: 52 passed.
- Clean BF16 warp-row extension smoke:
  `rm -rf /home/ubuntu/.cache/torch_extensions/py312_cu130/avo_cuda_warp_rows_attention_seed && AVO_VERBOSE_EXT_BUILD=1 uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Build output showed `/usr/local/cuda-12.9/bin/nvcc` with
    `-gencode=arch=compute_86,code=sm_86`.
  - Non-causal BF16: max abs error `0.00390625`, 0.911 ms.
  - Causal BF16: max abs error `0.015625`, 0.877 ms.
  - Geomean: `1.2962724058836346e-05` TFLOPS.
- FP32 warp-row smoke:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16 --dtype fp32 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Non-causal FP32: max abs error `4.76837158203125e-07`, 0.670 ms.
  - Causal FP32: max abs error `4.76837158203125e-07`, 0.856 ms.
  - Geomean: `1.5301870900236325e-05` TFLOPS.
- BF16 64x64 warp-row smoke:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens 64 --total-tokens 64 --num-heads 1 --head-dim 64 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Non-causal BF16: max abs error `0.00390625`, 0.702 ms.
  - Causal BF16: max abs error `0.0078125`, 0.622 ms.
  - Geomean: `0.0011223817750085961` TFLOPS.
- Same 64x64 BF16 shape on the previous tiled seed:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_tiled_attention_seed.py --seq-lens 64 --total-tokens 64 --num-heads 1 --head-dim 64 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Non-causal BF16: 1.026 ms.
  - Causal BF16: 0.923 ms.
  - Geomean: `0.000761717128185525` TFLOPS.
  - The warp-row seed is about 1.47x higher geomean on this noisy one-repeat tiny smoke.
- BF16 128x128 warp-row smoke:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens 128 --total-tokens 128 --num-heads 1 --head-dim 128 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Non-causal BF16: max abs error `0.00390625`, 0.965 ms.
  - Causal BF16: max abs error `0.0078125`, 0.852 ms.
  - Geomean: `0.006542664771912421` TFLOPS.
- Live Anthropic planner smoke:
  `uv run --extra agent python -m avo agent-plan --lineage /tmp/avo-agent-context-warp-rows-lineage --knowledge knowledge/ampere.md --cwd /home/ubuntu/avo-ampere --env-file /home/ubuntu/avo/.env.local`
  passed.
  - Returned `files_to_inspect`:
    - `candidates/cuda_warp_rows_attention_seed.py`;
    - `candidates/cuda_warp_rows_attention/attention_kernel.cu`;
    - `candidates/cuda_warp_rows_attention/attention.cpp`.
  - Returned bounded `next_command`:
    `avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`.

Tradeoffs and uncertainty:

- This is the first multi-row CTA seed, not a production attention kernel. It still uses scalar
  dot products and scalar PV accumulation, so absolute TFLOPS remain tiny.
- The 64x64 comparison suggests the row-per-warp layout is a useful incremental direction, but
  the benchmark uses one repeat on tiny shapes. Treat the ratio as a development signal, not a
  stable performance claim.
- The next kernel step should probably introduce tensor-core fragments for QK or PV on a tiny
  fixed head dimension, while preserving the warp-row seed as a correctness fallback.

## 2026-05-08 - Checkpoint 2.4: packed dot-product loads in the warp-row seed

Success criteria for this checkpoint:

- Improve the current warp-row scalar seed without changing its public candidate interface or
  adding a new kernel family.
- Add a conservative packed-load dot-product path only for head dimensions divisible by 4.
- Preserve a scalar fallback for odd/non-divisible head dimensions.
- Verify the packed path on BF16 and FP32 smoke shapes.
- Verify the scalar fallback with a deliberately odd FP32 head dimension.

Implementation:

- Updated `/home/ubuntu/avo-ampere/candidates/cuda_warp_rows_attention/attention_kernel.cu`.
- Added `ScalarPack4<scalar_t>` and a `dot_product(...)` helper.
- The helper:
  - uses a 4-wide packed path when `head_dim % 4 == 0`;
  - falls back to the original scalar loop otherwise;
  - keeps FP32 accumulation.
- Left the online softmax, row-per-warp layout, causal masking, and output accumulation
  structure unchanged.
- Updated `/home/ubuntu/avo-ampere/knowledge/ampere.md` to record the packed-load path and
  its alignment caveat.

Online research notes:

- Exa research found NVIDIA guidance that vectorized CUDA loads/stores can reduce executed
  instructions and improve bandwidth use, but require correct alignment of both the base
  allocation and any offset pointer. This is why the implementation only takes the packed path
  when the head dimension is divisible by 4 and keeps the scalar fallback.
  Source: https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
- Exa research on global-memory access reinforced that coalesced sequential access across a
  warp remains critical. This checkpoint does not solve the larger Q/K/V staging problem; it
  only reduces per-lane dot-loop load instruction pressure before adding tensor-core complexity.
  Source: https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/

Verification:

- `uv run --extra dev ruff check .` in `/home/ubuntu/avo-ampere`: passed.
- `uv run --extra dev pytest` in `/home/ubuntu/avo-ampere`: 52 passed.
- Clean BF16 packed-path extension smoke:
  `rm -rf /home/ubuntu/.cache/torch_extensions/py312_cu130/avo_cuda_warp_rows_attention_seed && AVO_VERBOSE_EXT_BUILD=1 uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Build output showed `/usr/local/cuda-12.9/bin/nvcc` with
    `-gencode=arch=compute_86,code=sm_86`.
  - Non-causal BF16: max abs error `0.00390625`, 0.724 ms.
  - Causal BF16: max abs error `0.015625`, 0.731 ms.
  - Geomean: `1.59304346366579e-05` TFLOPS.
- FP32 scalar-fallback odd-head-dim smoke:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 18 --dtype fp32 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Non-causal FP32: max abs error `2.682209014892578e-07`, 1.044 ms.
  - Causal FP32: max abs error `2.384185791015625e-07`, 0.841 ms.
  - Geomean: `1.3910434145795339e-05` TFLOPS.
- BF16 64x64 packed-path smoke:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens 64 --total-tokens 64 --num-heads 1 --head-dim 64 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Non-causal BF16: max abs error `0.00390625`, 0.618 ms.
  - Causal BF16: max abs error `0.0078125`, 0.702 ms.
  - Geomean: `0.00112539256257317` TFLOPS.
- BF16 128x128 packed-path smoke:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens 128 --total-tokens 128 --num-heads 1 --head-dim 128 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Non-causal BF16: max abs error `0.00390625`, 0.702 ms.
  - Causal BF16: max abs error `0.0078125`, 0.613 ms.
  - Geomean: `0.009043551093999186` TFLOPS.

Tradeoffs and uncertainty:

- This is still a scalar attention seed. Packed Q/K loads reduce loop/load overhead but do not
  address the main missing pieces: tensor-core QK/PV, shared-memory staging, and async copies.
- The speed deltas are one-repeat tiny-shape development signals. They are useful for steering
  the next edit, but not strong enough to claim stable performance improvement.

## 2026-05-08 - Checkpoint 2.5: gated shared K/V staging in the warp-row seed

Success criteria for this checkpoint:

- Add shared-memory K/V tile staging to the current warp-row seed without changing the candidate
  interface.
- Avoid deadlocks with causal rows that have different K limits inside one CTA.
- Preserve a global-memory fallback for boundary CTAs and larger head dimensions.
- Keep correctness against PyTorch SDPA for BF16 and FP32.
- Keep dispatch limited to the scorer-supported dtypes after shared-memory growth.

Implementation:

- Updated `/home/ubuntu/avo-ampere/candidates/cuda_warp_rows_attention/attention_kernel.cu`.
- Added per-CTA shared K and V tiles:
  - `k_tiles[32][128]`;
  - `v_tiles[32][128]`.
- Shared staging is enabled only when:
  - `head_dim <= 64`;
  - the CTA's four query rows stay inside the same sequence/head boundary.
- Head dimension 128 and boundary CTAs keep the previous global packed path.
- The shared path uses a block-wide K loop limit so every warp participates in `__syncthreads()`,
  even when causal masking means some rows have no valid keys in later tiles.
- Restricted CUDA dispatch to fp32, fp16, and bf16.

Important failure and fix:

- The first clean build failed because the old `AT_DISPATCH_FLOATING_TYPES_AND2` instantiated a
  double-precision kernel. With static shared K/V tiles, the double instantiation requested
  `0x10200` bytes of shared memory, above the default `0xc000` per-block limit.
- The scorer CLI only supports `bf16`, `fp16`, and `fp32`, so I replaced the broad dispatch macro
  with an explicit switch for those dtypes.

Online research notes:

- Exa research on NVIDIA's FlashAttention CUDA Tile post reinforced that the real FlashAttention
  direction is tiled K/V streaming through on-chip memory, online softmax state, and fused
  compute. This checkpoint implements only the K/V staging portion in the scalar seed.
  Source: https://developer.nvidia.com/blog/tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile/
- Exa research on line-by-line FlashAttention walkthroughs showed the common shared-memory
  partitioning pattern for `Qi`, `Kj`, `Vj`, scores, and running softmax state. I kept this
  checkpoint narrower by staging only K/V, leaving Q and O in registers/global reads.
  Source: https://www.stephendiehl.com/posts/flash_attention/

Verification:

- `uv run --extra dev ruff check .` in `/home/ubuntu/avo-ampere`: passed.
- `uv run --extra dev pytest` in `/home/ubuntu/avo-ampere`: 52 passed.
- Clean BF16 shared-path extension smoke:
  `rm -rf /home/ubuntu/.cache/torch_extensions/py312_cu130/avo_cuda_warp_rows_attention_seed && AVO_VERBOSE_EXT_BUILD=1 uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Build output showed `/usr/local/cuda-12.9/bin/nvcc` with
    `-gencode=arch=compute_86,code=sm_86`.
  - Non-causal BF16: max abs error `0.00390625`, 0.901 ms.
  - Causal BF16: max abs error `0.015625`, 0.868 ms.
  - Geomean: `1.3096654099117944e-05` TFLOPS.
- FP32 odd-shape fallback smoke:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens 18 --total-tokens 18 --num-heads 1 --head-dim 18 --dtype fp32 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Non-causal FP32: max abs error `4.172325134277344e-07`, 0.925 ms.
  - Causal FP32: max abs error `4.172325134277344e-07`, 0.696 ms.
  - Geomean: `2.055352117728248e-05` TFLOPS.
- BF16 64x64 shared-path smoke:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens 64 --total-tokens 64 --num-heads 1 --head-dim 64 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Non-causal BF16: max abs error `0.00390625`, 0.689 ms.
  - Causal BF16: max abs error `0.0078125`, 0.562 ms.
  - Geomean: `0.0011906303575200893` TFLOPS.
- BF16 128x128 global-path smoke:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens 128 --total-tokens 128 --num-heads 1 --head-dim 128 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Non-causal BF16: max abs error `0.00390625`, 0.719 ms.
  - Causal BF16: max abs error `0.0078125`, 0.547 ms.
  - Geomean: `0.009460695342707354` TFLOPS.

Tradeoffs and uncertainty:

- Shared K/V staging helped the 64x64 smoke in this run but regressed 128x128 before gating, so
  the implementation now stages only `head_dim <= 64`. This is a heuristic based on tiny
  one-repeat measurements, not a stable architecture conclusion.
- This still does not use tensor cores or async copy. It establishes a safer place to introduce
  `cp.async` later because the synchronization and boundary behavior are now explicit.

## 2026-05-08 - Checkpoint 2.6: cp.async staging attempt, not retained

Success criteria for this checkpoint:

- Try the smallest Ampere async-copy staging path on the current warp-row seed.
- Verify whether the compiled binary actually contains async copy instructions.
- Keep correctness gates authoritative.
- Do not retain the change if it regresses the tiny development smokes.
- Record the result so the agent does not rediscover the same dead-end immediately.

Implementation attempted:

- Temporarily updated
  `/home/ubuntu/avo-ampere/candidates/cuda_warp_rows_attention/attention_kernel.cu`.
- Added inline PTX helpers for:
  - `cp.async.cg.shared.global`;
  - `cp.async.commit_group`;
  - `cp.async.wait_group 0`.
- Added a 16-byte async copy path for staged K/V tiles when:
  - `head_dim <= 64`;
  - the CTA rows stayed inside one sequence/head;
  - `head_dim` was aligned to `16 / sizeof(dtype)` elements.
- Kept the synchronous staging path as fallback.

Outcome:

- The async-copy path compiled and passed correctness.
- `cuobjdump --dump-sass` on
  `/home/ubuntu/.cache/torch_extensions/py312_cu130/avo_cuda_warp_rows_attention_seed/avo_cuda_warp_rows_attention_seed.so`
  showed `LDGSTS.E.BYPASS.128`, `LDGDEPBAR`, and `DEPBAR`, confirming the inline PTX lowered to
  sm86 async-copy instructions.
- The 64x64 BF16 smoke regressed versus the synchronous staged version, so I removed the code
  and left the implementation at checkpoint 2.5 behavior.
- Updated `/home/ubuntu/avo-ampere/knowledge/ampere.md` with the negative result.

Online research notes:

- Exa research on CUTLASS `copy_sm80.hpp` showed the relevant low-level pattern:
  `cp.async.*.shared.global`, `cp.async.commit_group`, and `cp.async.wait_group`/`wait_all`.
  This matched the temporary implementation.
  Source: https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm80.hpp
- Exa research on CUTLASS `memory_sm80.h` also showed predicate-guarded and zero-fill variants,
  which are likely needed for a production boundary-safe path. I avoided adding those here and
  kept a simpler aligned-only experiment.
  Source: https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/memory_sm80.h

Verification:

- Clean BF16 async-copy extension smoke:
  `rm -rf /home/ubuntu/.cache/torch_extensions/py312_cu130/avo_cuda_warp_rows_attention_seed && AVO_VERBOSE_EXT_BUILD=1 uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed while the temporary async code was present.
  - Build output showed `/usr/local/cuda-12.9/bin/nvcc` with
    `-gencode=arch=compute_86,code=sm_86`.
  - Non-causal BF16: max abs error `0.00390625`, 0.837 ms.
  - Causal BF16: max abs error `0.015625`, 0.899 ms.
  - Geomean: `1.3351682505978846e-05` TFLOPS.
- SASS check:
  `cuobjdump --dump-sass /home/ubuntu/.cache/torch_extensions/py312_cu130/avo_cuda_warp_rows_attention_seed/avo_cuda_warp_rows_attention_seed.so | rg -n "CP_ASYNC|LDGSTS|DEPBAR|cp.async"`
  passed and found multiple `LDGSTS.E.BYPASS.128` instructions plus dependency barriers.
- FP32 odd-shape fallback smoke with the temporary async code:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens 18 --total-tokens 18 --num-heads 1 --head-dim 18 --dtype fp32 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Non-causal FP32: max abs error `4.172325134277344e-07`, 0.803 ms.
  - Causal FP32: max abs error `4.172325134277344e-07`, 0.857 ms.
  - Geomean: `1.988299258849849e-05` TFLOPS.
- BF16 64x64 async-copy smoke:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens 64 --total-tokens 64 --num-heads 1 --head-dim 64 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed correctness but regressed.
  - Non-causal BF16: max abs error `0.00390625`, 0.847 ms.
  - Causal BF16: max abs error `0.0078125`, 0.806 ms.
  - Geomean: `0.0008974849432840806` TFLOPS.
  - Previous synchronous staged checkpoint on the same shape recorded `0.0011906303575200893`
    TFLOPS geomean.

Tradeoffs and decision:

- The temporary async-copy path did not overlap copy with compute; it only replaced synchronous
  loads with async instructions followed immediately by wait. The regression is therefore not
  surprising.
- I did not keep the code because it worsened the development smoke and increased complexity.
  A future cp.async attempt should use double-buffering or a pipeline that overlaps K/V tile
  movement with score/PV work, and should include profiler evidence rather than only timing.

## 2026-05-08 - Checkpoint 2.7: replicate timing statistics for score records

Success criteria for this checkpoint:

- Improve the measurement substrate without changing any CUDA kernel behavior.
- Add a bounded way to collect repeated CUDA-event timings per case.
- Keep the existing one-trial score behavior compatible.
- Make timing noise visible in score JSON so the lineage gate is not relying on an opaque single
  average.
- Verify the change with unit tests and a tiny A6000 candidate smoke.

Implementation:

- Updated `/home/ubuntu/avo-ampere/avo/benchmark.py`.
- Added optional `trials` plumbing to `score_backend`.
- Added `timing_summary(samples_ms)` with:
  - raw `samples_ms`;
  - `trials`;
  - `min_ms`;
  - `median_ms`;
  - `mean_ms`;
  - coefficient of variation `cv`.
- Changed per-case `milliseconds` to use the median timing sample when multiple trials are
  requested. `tflops` is still computed from `milliseconds`, so the lineage-facing
  `geomean_tflops` now uses median timing in multi-trial runs.
- Updated `/home/ubuntu/avo-ampere/avo/cli.py` so `score`, `worker-score`, and `seed-baseline`
  accept and forward `--trials`.
- Updated `/home/ubuntu/avo-ampere/tests/test_benchmark_math.py` and
  `/home/ubuntu/avo-ampere/tests/test_cli.py`.
- Updated `/home/ubuntu/avo-ampere/README.md` and
  `/home/ubuntu/avo-ampere/knowledge/ampere.md`.

Online research notes:

- Exa search surfaced the PyTorch benchmark guidance that warmups, replicate measurements, and
  median statistics are important for noisy CUDA/kernel measurements. This directly motivated
  median-based score timing rather than adding another single average.
  Source: https://docs.pytorch.org/docs/stable/benchmark_utils.html
- Exa search also surfaced Lawrence Atkins' CUDA timing writeup, reinforcing the existing use of
  CUDA events and warmups for GPU timing.
  Source: https://lawrencium77.github.io/pytorch/cuda/gpu/2023/03/28/timings.html
- Exa search found Ampere FlashAttention examples and from-scratch worklogs that use profiling
  and repeated measurements as part of kernel optimization. This checkpoint keeps the scope to
  score JSON instrumentation rather than adding Nsight Compute integration yet.
  Sources:
  - https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py
  - https://github.com/rishisankar/flashattention2

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/benchmark.py avo/cli.py tests/test_benchmark_math.py tests/test_cli.py`
  passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_benchmark_math.py tests/test_cli.py tests/test_candidate_backend.py`
  passed, 11 tests.
- Full unit suite:
  `uv run --extra dev pytest`
  passed, 55 tests.
- Tiny CUDA candidate smoke:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16 --dtype bf16 --causal both --repeats 1 --warmup 1 --trials 3 --timeout-s 300`
  passed with `all_correct=true`.
  - Non-causal BF16 max abs error: `0.00390625`.
  - Non-causal samples: `[3.2025599479675293, 0.8102080225944519, 0.7377600073814392]` ms.
  - Non-causal median: `0.8102080225944519` ms.
  - Non-causal CV: `0.7232187690607705`.
  - Causal BF16 max abs error: `0.015625`.
  - Causal samples: `[0.7121279835700989, 0.7498559951782227, 0.8896639943122864]` ms.
  - Causal median: `0.7498559951782227` ms.
  - Causal CV: `0.09742281126662047`.
  - Geomean: `1.4863385361021696e-05` TFLOPS.

Tradeoffs and decision:

- This does not make the benchmark statistically complete. It gives the agent and lineage gate
  enough evidence to see when a run is noisy and to use a median sample instead of a single opaque
  timing value.
- The high non-causal CV in the smoke confirms why this is useful. Future larger comparisons
  should use more warmup and more trials, and eventually profiler counters, before accepting small
  performance deltas as real.

## 2026-05-08 - Checkpoint 2.8: reject degenerate lineage score payloads

Success criteria for this checkpoint:

- Tighten the lineage gate without changing candidate scoring or CUDA kernels.
- Prevent malformed or degenerate score records from becoming accepted lineage commits.
- Keep valid non-empty score payloads accepted when they pass correctness and throughput gates.
- Verify with focused gate tests and the full unit suite.

Implementation:

- Updated `/home/ubuntu/avo-ampere/avo/lineage.py`.
- `decide_gate` now rejects a candidate before throughput comparison when:
  - `all_correct` is false;
  - `cases` is missing, not a list, or empty;
  - `geomean_tflops` is non-positive or non-finite.
- Updated `/home/ubuntu/avo-ampere/tests/test_lineage.py` to cover empty cases, zero geomean,
  and infinite geomean.
- Updated `/home/ubuntu/avo-ampere/tests/test_evolve.py` mock accepted scores so they represent
  the real score shape more closely with a non-empty `cases` list.
- Updated `/home/ubuntu/avo-ampere/knowledge/ampere.md` gate notes.

Online research notes:

- Exa search on evolutionary/code-agent gating surfaced a common pattern: correctness/reviewer
  results are hard eligibility gates, and score selection happens after those gates. The local
  change keeps that shape by treating malformed score structure and degenerate throughput as hard
  eligibility failures before comparing geomean.
  Source: https://arxiv.org/pdf/2604.01210
- The same search also surfaced noisy-selection discussions in evolutionary systems. I did not add
  Elo, tournament selection, or population-level logic here because the current architecture is a
  single-lineage AVO loop; this checkpoint only hardens the existing single-lineage gate.
  Source: https://www.arxiv.org/pdf/2604.04347

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/lineage.py tests/test_lineage.py tests/test_evolve.py`
  passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_lineage.py tests/test_evolve.py`
  passed, 16 tests.
- Full lint:
  `uv run --extra dev ruff check .`
  passed.
- Full unit suite:
  `uv run --extra dev pytest`
  passed, 57 tests.
- Whitespace:
  `git diff --check` in `/home/ubuntu/avo-ampere` passed.

Tradeoffs and decision:

- This is deliberately stricter than the previous gate. Programmatic tests that used empty
  `cases` as placeholder accepted scores now need a non-empty list, which better matches actual
  score output.
- The gate still does not enforce a minimum relative improvement above measurement noise. That is
  a separate policy choice; the current change only prevents clearly invalid score payloads from
  entering lineage.

## 2026-05-08 - Checkpoint 2.9: rejected-attempt memory for agent prompts

Success criteria for this checkpoint:

- Preserve failed and rejected AVO attempts outside committed lineage.
- Feed recent attempt summaries back into `agent-plan` and `evolve-once` prompts.
- Keep the committed lineage limited to accepted score payloads.
- Avoid storing huge stdout/stderr blobs in prompt context.
- Verify with unit tests and lint.

Implementation:

- Updated `/home/ubuntu/avo-ampere/avo/evolve.py`.
- Added `write_step_record(directory, step)`:
  - writes a timestamped `EvolutionStep` JSON file;
  - avoids overwriting on timestamp collision by adding a numeric suffix.
- Added `summarize_attempt_history(directory, limit=5)`:
  - reads recent `*.json` step records;
  - skips malformed JSON files;
  - emits compact lines with command status, gate status, score status, command, and hypothesis;
  - intentionally excludes long stdout/stderr tails from the summary.
- Updated `/home/ubuntu/avo-ampere/avo/agent.py`:
  - `build_variation_prompt` accepts `attempt_history`;
  - `request_variation_decision` passes it through;
  - prompts now tell the agent to avoid repeating failed or regressed directions.
- Updated `/home/ubuntu/avo-ampere/avo/cli.py`:
  - `agent-plan` and `evolve-once` accept `--attempts-dir` and `--attempt-limit`;
  - `evolve-once --attempts-dir DIR` reads recent history before planning and writes the new
    step after finalizing.
- Updated `/home/ubuntu/avo-ampere/tests/test_agent.py` and
  `/home/ubuntu/avo-ampere/tests/test_evolve.py`.
- Updated `/home/ubuntu/avo-ampere/README.md` and
  `/home/ubuntu/avo-ampere/knowledge/ampere.md`.

Online and local research notes:

- Exa search on the AVO paper reinforced that unsuccessful intermediate attempts are part of the
  agent's internal search trajectory, while only successful candidates become committed lineage.
  This checkpoint adds a durable local equivalent: attempt records outside lineage.
  Source: https://arxiv.org/html/2603.24517
- Exa search on Anthropic tool-use docs reinforced the importance of strict tool schemas and
  high-signal tool results. The attempt summary deliberately reports only stable fields needed for
  the next decision instead of dumping entire logs into prompt context.
  Sources:
  - https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use
  - https://console.anthropic.com/docs/en/agents-and-tools/tool-use/handle-tool-calls
- I inspected the local `pi-mono-agent/packages/agent` High-agent implementation for patterns only.
  The useful pattern was durable, ordered tool result/state handling plus context transformation for
  pruning/compaction. I did not copy its TypeScript implementation or event loop.

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/agent.py avo/cli.py avo/evolve.py tests/test_agent.py tests/test_evolve.py`
  passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_agent.py tests/test_evolve.py tests/test_cli.py`
  passed, 35 tests.
- Full lint:
  `uv run --extra dev ruff check .`
  passed.
- Full unit suite:
  `uv run --extra dev pytest`
  passed, 60 tests.
- CLI parser checks:
  `uv run python -m avo agent-plan --help` and
  `uv run python -m avo evolve-once --help`
  both showed `--attempts-dir` and `--attempt-limit`.
- Whitespace:
  `git diff --check` in `/home/ubuntu/avo-ampere` and `/home/ubuntu/avo` passed.

Tradeoffs and decision:

- This still is not a full edit-evaluate-debug mutation loop. It makes the single-step planner less
  forgetful by giving it recent rejected attempts, while keeping the actual shell execution surface
  bounded to existing `avo` commands.
- The attempt summaries are intentionally lossy. Exact logs remain in JSON files, while prompt
  context gets only the signal needed to avoid repeated dead ends.

## 2026-05-08 - Checkpoint 3.0: Anthropic readiness in env status

Success criteria for this checkpoint:

- Make the Anthropic live-planner prerequisite visible before running `agent-plan` or
  `evolve-once`.
- Keep the check in the existing CLI surface instead of adding a new subsystem.
- Do not print or persist the Anthropic API key value.
- Allow the status check to load a local env file because the current shell does not have
  `ANTHROPIC_API_KEY` set.
- Verify with unit tests, CLI smoke checks, and lint.

Implementation:

- Updated `/home/ubuntu/avo-ampere/avo/cli.py`.
- `avo env` now accepts `--env-file PATH`.
- The existing environment JSON now includes an `agent` block with:
  - whether the Anthropic Python SDK can be imported;
  - the Anthropic SDK version when available;
  - whether `ANTHROPIC_API_KEY` is present;
  - whether the requested env file existed and was loaded.
- Updated `/home/ubuntu/avo-ampere/tests/test_cli.py` so the agent status reports key presence
  without including the secret value.
- Updated `/home/ubuntu/avo-ampere/README.md` with the readiness command.

Online research notes:

- Exa search found Anthropic's official Python SDK docs showing `Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))`
  and dotenv-style local key loading as the documented path. The runtime already used this shape;
  this checkpoint exposes it in `avo env` so a missing key is visible before a live planner call.
  Source: https://console.anthropic.com/docs/en/api/sdks/python
- Exa search found Anthropic key-handling guidance recommending environment variables and keeping
  local dotenv files out of source control. The readiness JSON reports only booleans and metadata,
  not the key value.
  Source: https://support.anthropic.com/en/articles/9767949-api-key-best-practices-keeping-your-keys-safe-and-secure

Local finding:

- The current shell still reports `ANTHROPIC_API_KEY` as missing.
- `/home/ubuntu/avo/.env.local` exists, while `/home/ubuntu/avo-ampere` has no local env file.
- `uv run --extra agent ... import anthropic` reported version `0.100.0` before this checkpoint.

Tradeoffs and decision:

- This is a readiness checkpoint, not a live Anthropic planner run. It removes ambiguity about why
  a planner smoke can fail, but the actual `agent-plan` call still requires a valid key in the
  process environment or a supplied env file.

## 2026-05-08 - Checkpoint 3.1: live evolve-once smoke in throwaway lineage

Success criteria for this checkpoint:

- Exercise the current live Anthropic planner with `--env-file ../avo/.env.local`.
- Keep the run isolated from the real runtime repo and real lineage state.
- Verify the strict-tool decision can drive the bounded command executor.
- Verify score extraction, correctness gating, temp lineage commit, and attempts-dir persistence.
- Avoid code changes unless the live path exposes a reliability bug.

Online research notes:

- Exa search found Anthropic's current strict tool-use guidance: setting `strict: true` on the tool
  definition constrains tool `input` to the schema and returns validated data in the `tool_use`
  block. This is the exact contract the AVO planner is using for variation decisions.
  Source: https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use
- Exa search found Anthropic's parallel tool-use docs and troubleshooting notes confirming that
  `disable_parallel_tool_use=true` with a specific tool choice should produce exactly one tool call
  for the request. That supports the current single-decision planner boundary.
  Sources:
  - https://console.anthropic.com/docs/en/agents-and-tools/tool-use/parallel-tool-use
  - https://console.anthropic.com/docs/en/agents-and-tools/tool-use/troubleshooting-tool-use

Commands and results:

- Live planner smoke:
  `uv run --extra agent python -m avo agent-plan --lineage /tmp/avo-agent-live-lineage --knowledge knowledge/ampere.md --attempts-dir /tmp/avo-agent-live-attempts --env-file ../avo/.env.local`
  passed and returned a validated JSON decision.
- The decision used existing local files only:
  - `candidates/cuda_warp_rows_attention_seed.py`;
  - `candidates/cuda_warp_rows_attention/attention_kernel.cu`;
  - `candidates/cuda_warp_rows_attention/attention.cpp`.
- The bounded `next_command` was:
  `avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`.
- Live one-step smoke:
  `uv run --extra agent --extra cuda python -m avo evolve-once --lineage /tmp/avo-agent-live-lineage --knowledge knowledge/ampere.md --attempts-dir /tmp/avo-agent-live-attempts --step-json /tmp/avo-agent-live-step.json --env-file ../avo/.env.local --timeout-s 300`
  passed.
- The bounded score command returned `all_correct: true`.
  - Non-causal BF16 smoke: max abs error `0.00390625`, `0.7721279859542847` ms,
    `2.121928009091753e-05` TFLOPS.
  - Causal BF16 smoke: max abs error `0.015625`, `0.7422080039978027` ms,
    `1.1037337182939155e-05` TFLOPS.
  - Geomean: `1.5303736443845484e-05` TFLOPS.
- The temp lineage at `/tmp/avo-agent-live-lineage` was initialized and accepted one candidate
  commit because the best previous geomean was `0.0`.
- The attempts directory wrote
  `/tmp/avo-agent-live-attempts/2026-05-08T03-54-12-00-00.json`.

Tradeoffs and decision:

- This verifies the live agent-orchestrator path, but it does not represent a new real lineage
  improvement. The run used a throwaway lineage and repeated the tiny warp-row smoke as a safe
  end-to-end check.
- No runtime code change was needed. The live agent still selected a conservative verification
  command, which is acceptable for an empty temporary lineage but shows the next real progress
  still needs a safe edit/evaluate mutation path or a tensor-core candidate direction.

## 2026-05-08 - Checkpoint 3.2: tiny BF16 WMMA QK attention seed

Success criteria for this checkpoint:

- Add the first local tensor-core attention candidate without attempting a production rewrite.
- Keep the shape fixed and tiny enough to debug: BF16, sequence length 16, head dimension 16.
- Use Ampere-compatible CUDA WMMA/BF16 on `sm_86`; do not introduce Blackwell-only assumptions.
- Keep PyTorch SDPA as the correctness reference through the existing candidate scorer.
- Verify that generated SASS contains tensor-core instructions, not only scalar code.

Implementation:

- Added `/home/ubuntu/avo-ampere/candidates/cuda_mma_attention_seed.py`.
- Added `/home/ubuntu/avo-ampere/candidates/cuda_mma_attention/attention.cpp`.
- Added `/home/ubuntu/avo-ampere/candidates/cuda_mma_attention/attention_kernel.cu`.
- The CUDA kernel handles one `(batch, head)` block for a fixed 16x16 tile:
  - warp 0 computes the full QK score tile with CUDA WMMA BF16 inputs and FP32 accumulators;
  - the block stores scores to shared memory;
  - scalar code applies causal/non-causal softmax and multiplies by V;
  - output is written as BF16.
- Updated `/home/ubuntu/avo-ampere/avo/agent.py` so repo context prefers the new MMA seed for
  first local candidate scoring.
- Updated `/home/ubuntu/avo-ampere/tests/test_agent.py`, `README.md`, and `knowledge/ampere.md`.

Online research notes:

- Exa search found CUTLASS documentation stating that TensorOp BF16 multiply with FP32
  accumulation is supported for compute capability 80+ and CUDA 11.4+, and that CUDA exposes
  tensor cores through the `nvcuda::wmma` API. This supports using WMMA on the local sm86 target.
  Sources:
  - https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/functionality.md
  - https://docs.nvidia.com/cutlass/4.4.0/media/docs/cpp/functionality.html
- Exa search found CUTLASS Ampere examples using BF16 inputs and FP32 accumulators on Ampere
  tensor cores. I used these only as architectural confirmation; the local code is a small
  standalone PyTorch extension and does not copy CUTLASS code.
  Source: https://github.com/NVIDIA/cutlass/blob/d4e16f5d/examples/23_ampere_gemm_operand_reduction_fusion/ampere_gemm_operand_reduction_fusion.cu

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/agent.py tests/test_agent.py candidates/cuda_mma_attention_seed.py`
  passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_agent.py tests/test_candidate_backend.py`
  passed, 25 tests.
- Clean CUDA extension smoke:
  `rm -rf /home/ubuntu/.cache/torch_extensions/py312_cu130/avo_cuda_mma_attention_seed && AVO_VERBOSE_EXT_BUILD=1 uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_mma_attention_seed.py --seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Build output showed `/usr/local/cuda-12.9/bin/nvcc` with
    `-gencode=arch=compute_86,code=sm_86`.
  - Non-causal BF16: max abs error `0.00390625`, `0.8457919955253601` ms,
    `1.937119301989037e-05` TFLOPS.
  - Causal BF16: max abs error `0.015625`, `1.2148799896240234` ms,
    `6.7430528693910166e-06` TFLOPS.
  - Geomean: `1.1428953524986394e-05` TFLOPS.
- SASS check:
  `cuobjdump --dump-sass /home/ubuntu/.cache/torch_extensions/py312_cu130/avo_cuda_mma_attention_seed/avo_cuda_mma_attention_seed.so | rg -n "HMMA|MMA|BMMA|wmma|WGMMA"`
  found `HMMA.16816.F32.BF16`.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 62 tests.
- Whitespace:
  `git diff --check` in `/home/ubuntu/avo-ampere` passed.

Tradeoffs and decision:

- This candidate is intentionally not a performance result. It uses tensor cores only for QK,
  while softmax and PV remain scalar. It also only supports the fixed 16x16 BF16 smoke shape.
- The value is that the repo now has a real tensor-core attention foothold on sm86. The next
  meaningful kernel work is either tensor-core PV on the same fixed shape or tiling this QK path
  beyond 16 keys while preserving the existing correctness gate.

## 2026-05-08 - Checkpoint 3.3: add tensor-core PV to the tiny MMA seed

Success criteria for this checkpoint:

- Extend the fixed 16x16 BF16 MMA seed from QK-only tensor-core work to QK plus PV tensor-core work.
- Keep the same tiny smoke shape and avoid widening the implementation before correctness is stable.
- Preserve PyTorch SDPA correctness under the existing BF16 tolerance despite storing softmax
  probabilities as BF16.
- Verify SASS still contains tensor-core instructions for the revised kernel.

Implementation:

- Updated `/home/ubuntu/avo-ampere/candidates/cuda_mma_attention/attention_kernel.cu`.
- The kernel now:
  - computes QK with WMMA BF16 inputs and FP32 accumulators;
  - computes row-wise softmax in scalar FP32;
  - stores the 16x16 probability tile as BF16 in shared memory;
  - computes PV with a second WMMA BF16 multiply and FP32 accumulator;
  - stores the final output as BF16.
- Updated `/home/ubuntu/avo-ampere/README.md` and
  `/home/ubuntu/avo-ampere/knowledge/ampere.md` from QK wording to QK/PV wording.

Online research notes:

- Exa search found NVIDIA's CUTLASS Ampere FlashAttention v2 example describing the Ampere flow as
  QK MMA, softmax, then V/PV MMA with cp.async and register pipelining around it. This checkpoint
  implements only the smallest local correctness analogue of that two-MMA structure, not the full
  FA2 pipeline.
  Source: https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py
- Exa search also surfaced attention-kernel papers describing forward attention as QK followed by
  PV, with both as MMA operations around softmax. This reinforces that adding PV is the next
  correct tensor-core foothold after the QK seed.
  Sources:
  - https://browse.arxiv.org/html/2312.11918v1
  - https://www.arxiv.org/pdf/2603.05451

Verification:

- Clean CUDA extension smoke:
  `rm -rf /home/ubuntu/.cache/torch_extensions/py312_cu130/avo_cuda_mma_attention_seed && AVO_VERBOSE_EXT_BUILD=1 uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_mma_attention_seed.py --seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Build output showed `/usr/local/cuda-12.9/bin/nvcc` with
    `-gencode=arch=compute_86,code=sm_86`.
  - Non-causal BF16: max abs error `0.00390625`, `0.7331519722938538` ms,
    `2.2347344915050092e-05` TFLOPS.
  - Causal BF16: max abs error `0.015625`, `0.602944016456604` ms,
    `1.3586667711113453e-05` TFLOPS.
  - Geomean: `1.7424865841274845e-05` TFLOPS.
- SASS check:
  `cuobjdump --dump-sass /home/ubuntu/.cache/torch_extensions/py312_cu130/avo_cuda_mma_attention_seed/avo_cuda_mma_attention_seed.so | rg -n "HMMA|MMA|BMMA|wmma|WGMMA"`
  found four `HMMA.16816.F32.BF16` instructions, corresponding to the two WMMA calls.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 62 tests.
- Whitespace:
  `git diff --check` in `/home/ubuntu/avo-ampere` passed.

Tradeoffs and decision:

- Storing probabilities as BF16 is a precision compromise made solely to use WMMA PV through the
  public CUDA API in a tiny seed. It passed the current BF16 correctness gate, but future larger
  shapes may need a different representation or tolerance analysis.
- This is still not a production attention kernel. It proves the local repo can execute both
  attention MMAs on sm86 tensor cores under the existing scorer. The next meaningful direction is
  tiling beyond 16 keys or introducing online softmax state around multiple QK/PV tiles.

## 2026-05-08 - Checkpoint 3.4: two-tile online softmax in the MMA seed

Success criteria for this checkpoint:

- Extend the tiny MMA seed beyond a single 16-key tile without changing the production benchmark
  suite or claiming performance.
- Support both sequence length 16 and 32 with head dimension 16 and BF16.
- For sequence length 32, process two 16-key tiles with FP32 online softmax state:
  running row maximum, running denominator, and unnormalized output accumulator.
- Keep QK and PV on tensor cores for every tile.
- Verify causal and non-causal correctness against PyTorch SDPA for both 32-token and 16-token
  smokes.

Implementation:

- Updated `/home/ubuntu/avo-ampere/candidates/cuda_mma_attention/attention_kernel.cu`.
- The kernel now launches one CTA per `(batch, head, 16-query tile)`.
- Each CTA iterates over 16-key K/V tiles:
  - compute a 16x16 QK score tile with WMMA;
  - update per-row FP32 online softmax state;
  - store unnormalized softmax weights as BF16 in shared memory;
  - rescale the previous FP32 output accumulator;
  - compute the current PV tile with WMMA;
  - accumulate and normalize only after all K/V tiles finish.
- Updated `/home/ubuntu/avo-ampere/candidates/cuda_mma_attention_seed.py` to allow sequence
  lengths 16 and 32.
- Updated `/home/ubuntu/avo-ampere/avo/agent.py` so the preferred local MMA smoke command uses
  `--seq-lens 32 --total-tokens 32`, exercising the online path.
- Updated `/home/ubuntu/avo-ampere/tests/test_agent.py`, `README.md`, and `knowledge/ampere.md`.

Online research notes:

- Exa search on FlashAttention online softmax resurfaced the core recurrence:
  update the row max, rescale the old denominator/output accumulator, add the current tile's
  exponentials and PV contribution, and divide at the end. This is the recurrence implemented in
  the two-tile smoke.
  Sources:
  - https://arxiv.org/pdf/2205.14135
  - https://peterchng.com/blog/2024/06/26/the-basic-idea-behind-flashattention/
  - https://mbrenndoerfer.com/writing/flashattention-algorithm-memory-efficient-gpu-tiling

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/agent.py tests/test_agent.py candidates/cuda_mma_attention_seed.py`
  passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_agent.py tests/test_candidate_backend.py`
  passed, 25 tests.
- Clean two-tile CUDA extension smoke:
  `rm -rf /home/ubuntu/.cache/torch_extensions/py312_cu130/avo_cuda_mma_attention_seed && AVO_VERBOSE_EXT_BUILD=1 uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_mma_attention_seed.py --seq-lens 32 --total-tokens 32 --num-heads 1 --head-dim 16 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Build output showed `/usr/local/cuda-12.9/bin/nvcc` with
    `-gencode=arch=compute_86,code=sm_86`.
  - Non-causal BF16 32-token smoke: max abs error `0.001953125`,
    `0.7738879919052124` ms, `8.468408953944203e-05` TFLOPS.
  - Causal BF16 32-token smoke: max abs error `0.00390625`,
    `0.6883839964866638` ms, `4.7601339030598484e-05` TFLOPS.
  - Geomean: `6.34907556787957e-05` TFLOPS.
- SASS check:
  `cuobjdump --dump-sass /home/ubuntu/.cache/torch_extensions/py312_cu130/avo_cuda_mma_attention_seed/avo_cuda_mma_attention_seed.so | rg -n "HMMA|MMA|BMMA|wmma|WGMMA"`
  found `HMMA.16816.F32.BF16` instructions.
- Cached 16-token regression smoke:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_mma_attention_seed.py --seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed.
  - Non-causal BF16 16-token smoke: max abs error `0.0`.
  - Causal BF16 16-token smoke: max abs error `0.0`.
  - Geomean: `1.3579363584840928e-05` TFLOPS.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 62 tests.
- Whitespace:
  `git diff --check` in `/home/ubuntu/avo-ampere` passed.

Tradeoffs and decision:

- This is still a tiny correctness seed. It only reaches 32 tokens, still stores softmax weights as
  BF16 between softmax and PV, and has no cp.async pipeline or production shared-memory layout.
- The important progress is algorithmic: the local tensor-core candidate now crosses a tile
  boundary using online softmax state while remaining inside the existing isolated scorer and
  correctness gate.

## 2026-05-08 - Checkpoint 3.5: bounded candidate patch substrate

Success criteria for this checkpoint:

- Add a minimal file-edit executor needed for the future Anthropic mutation loop.
- Accept only ordinary unified diffs targeting candidate files under `candidates/`.
- Reject path traversal, non-candidate paths, symlink-mode patches, existing symlink paths, binary
  patches, renames, deletes, and mode changes before invoking Git.
- Run `git apply --check --whitespace=error` before any real apply.
- Do not stage, commit, score, or wire the command into the autonomous `run-decision` allowlist yet.

Implementation:

- Added `/home/ubuntu/avo-ampere/avo/evolve.py` patch primitives:
  - `PatchResult`;
  - `paths_from_unified_diff(...)`;
  - `apply_candidate_patch(...)`.
- Added `avo apply-patch PATCH [--cwd PATH] [--dry-run]`.
- Updated tests in `/home/ubuntu/avo-ampere/tests/test_evolve.py` and
  `/home/ubuntu/avo-ampere/tests/test_cli.py`.
- Updated `/home/ubuntu/avo-ampere/README.md` and
  `/home/ubuntu/avo-ampere/knowledge/ampere.md`.

Online research notes:

- Exa search found the Git `apply` documentation: `git apply --check` checks whether a patch
  applies without applying it, `git apply` rejects paths outside the working area by default, and
  the docs discourage context-free `--unidiff-zero` patches because normal unified context provides
  safety checks.
  Source: https://git-scm.com/docs/git-apply/2.51.0
- Exa search also found Git security advisory GHSA-r87m-v37r-cwfh / CVE-2023-23946, where crafted
  patches involving symlinks could overwrite outside the working tree. The local `git --version`
  reports `2.34.1`, which is in the affected version range, so this checkpoint rejects symlink
  creation and existing symlink paths before calling `git apply`.
  Source: https://github.com/git/git/security/advisories/GHSA-r87m-v37r-cwfh
- The resulting design treats Git's own checks as necessary but not sufficient on this machine.
  The allowlist parser is deliberately narrower than general `git apply`.

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/evolve.py avo/cli.py tests/test_evolve.py tests/test_cli.py`
  passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_evolve.py tests/test_cli.py`
  passed, 28 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 74 tests.
- Whitespace:
  `git diff --check` in `/home/ubuntu/avo-ampere` passed.

Tradeoffs and decision:

- `apply-patch` rejects useful but riskier patch forms such as renames, deletes, binary files,
  executable mode changes, and paths with whitespace. Candidate kernels do not need those forms
  for the next loop step.
- This checkpoint does not yet let the Anthropic agent emit or apply edits. It adds the bounded
  executor that a later schema-validated patch decision can call before scoring and lineage gating.

## 2026-05-08 - Checkpoint 3.6: schema-carried candidate patches

Success criteria for this checkpoint:

- Let the Anthropic variation decision carry one bounded candidate edit without widening the
  `next_command` shell/CLI allowlist.
- Keep `next_command` limited to `avo env`, `avo compile`, and `avo score`.
- Apply a non-empty `candidate_patch` through the existing `candidates/`-only patch validator
  before running the bounded command.
- Record patch success or rejection in attempt JSON and attempt-history summaries.
- Recover safely from malformed non-diff patch commentary in the agent response.
- Verify the real Anthropic planning path with `ANTHROPIC_API_KEY` loaded from `.env.local`.

Implementation:

- Updated `/home/ubuntu/avo-ampere/avo/agent.py`:
  - added required schema field `candidate_patch`;
  - raised the Anthropic response token cap from 1200 to 4000 for small diffs;
  - normalized non-diff patch commentary to an empty patch;
  - kept markdown-fenced diff content invalid;
  - added retry feedback for invalid structured decisions.
- Updated `/home/ubuntu/avo-ampere/avo/evolve.py`:
  - `run_decision_command(...)` applies `candidate_patch` before the command;
  - patch failures stop command execution;
  - `VariationAttempt` records `patch_result`;
  - attempt history now reports patch status.
- Updated `/home/ubuntu/avo-ampere/tests/test_agent.py` and
  `/home/ubuntu/avo-ampere/tests/test_evolve.py`.
- Updated `/home/ubuntu/avo-ampere/README.md` and
  `/home/ubuntu/avo-ampere/knowledge/ampere.md`.

Online research notes:

- Exa search found Anthropic's strict tool-use docs: strict tool definitions schema-constrain tool
  inputs and are recommended for agentic workflows. This supports keeping the decision as a strict
  tool input rather than free-form text.
  Source: https://console.anthropic.com/docs/en/agents-and-tools/tool-use/strict-tool-use
- Exa search found Anthropic structured-output docs noting schema-constrained output paths and the
  need for `additionalProperties: false`. It also surfaced the all-properties-required guidance for
  structured JSON outputs, so `candidate_patch` is required in the schema but defaults to empty when
  older saved decision JSON is loaded locally.
  Source: https://docs.claude.com/en/docs/build-with-claude/structured-outputs
- Exa search found Anthropic tool-use troubleshooting guidance: strict mode handles schema shape,
  but tool descriptions and examples still matter for ambiguous parameters. The live smoke confirmed
  the distinction: the first schema-valid response put non-diff commentary into `candidate_patch`.
  Source: https://console.anthropic.com/docs/en/agents-and-tools/tool-use/troubleshooting-tool-use

Verification:

- First live Anthropic smoke:
  `rm -rf /tmp/avo-agent-patch-lineage && uv run --extra agent --extra cuda python -m avo agent-plan --lineage /tmp/avo-agent-patch-lineage --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local`
  initially failed after three retries because `candidate_patch` was a string but not a diff. This
  exposed that strict schema alone was not enough semantic validation.
- After normalizing non-diff patch commentary to empty, the same live Anthropic smoke passed. The
  returned decision had `candidate_patch: ""` and the expected bounded score command for
  `candidates/cuda_mma_attention_seed.py` on the 32-token BF16 smoke.
- Focused lint:
  `uv run --extra dev ruff check avo/agent.py avo/evolve.py tests/test_agent.py tests/test_evolve.py`
  passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_agent.py tests/test_evolve.py tests/test_cli.py`
  passed, 56 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 81 tests.
- Whitespace:
  `git diff --check` in `/home/ubuntu/avo-ampere` passed.

Tradeoffs and decision:

- Non-diff `candidate_patch` text is treated as no patch instead of killing the planning step. This
  is conservative: it cannot apply an unintended edit, and the bounded score command can still run.
- A malformed diff-like patch is still rejected by the patch executor before command execution.
- This is still a one-step loop, not a full multi-session autonomous search. The next missing layer
  is repeated edit/score/diagnose control flow with cleanup policy for rejected working-tree edits.

## 2026-05-08 - Checkpoint 3.7: rejected patch cleanup

Success criteria for this checkpoint:

- Prevent schema-carried candidate patches from polluting the working tree after a failed or
  non-accepted `evolve-once` step.
- Keep accepted patched steps intact so the accepted source edit remains available for the next
  local candidate state.
- Record cleanup success or failure in step JSON and attempt-history summaries.
- Surface cleanup failure as a failed `evolve-once` result.
- Do not broaden the patch allowlist, command allowlist, or lineage gate.

Implementation:

- Added `/home/ubuntu/avo-ampere/avo/evolve.py` helper `revert_candidate_patch(...)`, which
  validates the same candidate-only patch paths and uses `git apply --reverse`.
- Added `cleanup_rejected_candidate_patch(...)`:
  - no-op for accepted steps;
  - no-op for missing or rejected patches that were never applied;
  - reverse-applies an applied patch when the step has no accepted gate decision.
- Extended `EvolutionStep` with `patch_cleanup_result`.
- Updated attempt-history summaries to include cleanup status.
- Updated `/home/ubuntu/avo-ampere/avo/cli.py` so `evolve-once` runs cleanup after finalizing
  the attempt and returns failure if cleanup itself fails.
- Updated `/home/ubuntu/avo-ampere/tests/test_evolve.py` and
  `/home/ubuntu/avo-ampere/tests/test_cli.py`.
- Updated `/home/ubuntu/avo-ampere/README.md` and
  `/home/ubuntu/avo-ampere/knowledge/ampere.md`.

Online research notes:

- Exa search found the Git `apply` documentation for reverse patching: `git apply --check` checks
  applicability without applying, `git apply --reverse` applies a patch in reverse, and default
  `git apply` behavior is atomic unless `--reject` is used. The cleanup path uses the same pattern:
  reverse check first, then reverse apply, with no `--reject`.
  Sources:
  - https://git-scm.com/docs/git-apply
  - https://code.googlesource.com/git/+/refs/tags/v2.34.8/Documentation/git-apply.txt

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/evolve.py avo/cli.py tests/test_evolve.py tests/test_cli.py`
  passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_evolve.py tests/test_cli.py`
  passed, 34 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 85 tests.
- Whitespace:
  `git diff --check` in `/home/ubuntu/avo-ampere` passed.

Tradeoffs and decision:

- Cleanup currently happens only in `evolve-once`, where the gate outcome is known. Manual
  `run-decision` still leaves a successful patch in place because it has no lineage gate context.
- Reverse cleanup can fail if something else mutates the same candidate paths between patch apply
  and cleanup. In that case `evolve-once` exits non-zero and records the cleanup failure instead of
  silently hiding a dirty state.
- This still does not solve source persistence for accepted candidates in a separate lineage repo.
  It only makes rejected single-step edits safe enough to continue iterating.

## 2026-05-08 - Checkpoint 3.8: accepted patch source snapshots

Success criteria for this checkpoint:

- Persist accepted schema-carried candidate edits into the lineage repository, not only the runtime
  working tree.
- Keep source persistence behind the existing correctness/throughput gate: rejected candidates must
  not create lineage source artifacts.
- Store enough source context to inspect the accepted patch without broadening the runtime repo's
  commit surface.
- Keep the implementation narrow: store the raw accepted patch and snapshots of the candidate files
  touched by that patch.
- Preserve the existing rejected-patch cleanup behavior.

Implementation:

- Updated `/home/ubuntu/avo-ampere/avo/lineage.py`:
  - `commit_score(...)` now accepts optional `source_files` and `candidate_patch`;
  - accepted commits write `sources/latest/<repo-relative-candidate-path>` and
    `patches/latest.patch` alongside `scores/latest.json`;
  - source paths are validated to stay under `candidates/`;
  - rejected candidates return before writing source artifacts or advancing HEAD.
- Updated `/home/ubuntu/avo-ampere/avo/evolve.py`:
  - `finalize_attempt(..., source_root=...)` snapshots touched candidate files when a patch was
    applied successfully;
  - unpatched attempts still commit only scores.
- Updated `/home/ubuntu/avo-ampere/avo/cli.py` so `evolve-once` passes its `--cwd` as the source
  root for accepted source snapshots.
- Updated tests in `/home/ubuntu/avo-ampere/tests/test_lineage.py`,
  `/home/ubuntu/avo-ampere/tests/test_evolve.py`, and
  `/home/ubuntu/avo-ampere/tests/test_cli.py`.
- Updated `/home/ubuntu/avo-ampere/README.md` and
  `/home/ubuntu/avo-ampere/knowledge/ampere.md`.

Online research notes:

- Exa search found EvoGit's Git-based code evolution paper, which records each concrete code
  version as a Git commit for transparency, reproducibility, and replay of the trajectory. This
  supports moving beyond score-only commits in the AVO lineage.
  Source: https://arxiv.org/html/2506.02049v1
- Exa search found program-repair work discussing Git as an efficient way to store APR histories
  because evolutionary source changes are small and Git preserves modifications compactly. This
  supports storing accepted patches and touched source snapshots instead of ad hoc text logs.
  Source: https://sdl.ist.osaka-u.ac.jp/pman/pman3.cgi?DOWNLOAD=526
- Exa search also surfaced EvoRepair's output structure, which separates high-quality accepted
  patches from rejected/valid intermediate patches. This matches the current design: accepted source
  artifacts go into lineage; rejected attempts stay in attempt history.
  Source: https://github.com/nus-apr/evoRepair

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/lineage.py avo/evolve.py avo/cli.py tests/test_lineage.py tests/test_evolve.py tests/test_cli.py`
  passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_lineage.py tests/test_evolve.py tests/test_cli.py`
  passed, 44 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 89 tests.
- Whitespace:
  `git diff --check` in `/home/ubuntu/avo-ampere` passed.

Tradeoffs and decision:

- The snapshot currently includes only files touched by the accepted patch, plus the raw patch. This
  is enough to inspect and replay the accepted edit, but it is not yet a complete candidate tree
  snapshot when unchanged companion files matter.
- Direct `commit-score` remains score-only because it has no source root or candidate patch context.
- This moves the lineage closer to the architecture writeup's `(source, score)` requirement, but a
  complete multi-step autonomous search loop is still missing.

## 2026-05-08 - Checkpoint 3.9: bounded evolve loop

Success criteria for this checkpoint:

- Add a minimal multi-step autonomous driver without expanding the single-step safety surface.
- Keep a hard loop cap so the agent cannot run indefinitely.
- Require attempt-history storage so each step can see recent accepted, rejected, and failed
  attempts.
- Stop on accepted candidates, cleanup failure, or max-step exhaustion.
- Preserve the existing bounded command allowlist, candidate-patch validator, lineage gate, and
  rejected-patch cleanup behavior.

Implementation:

- Updated `/home/ubuntu/avo-ampere/avo/cli.py`:
  - added `avo evolve-loop`;
  - refactored `evolve-once` through a shared `_run_evolve_step(...)` helper;
  - required `--attempts-dir` for `evolve-loop` cross-step memory;
  - added `--max-steps` with a positive-value guard and default of 3;
  - added optional `--loop-json` summary output;
  - repeated the existing request, bounded command, gate, and cleanup unit until accepted,
    cleanup-failed, or max-steps.
- Updated `/home/ubuntu/avo-ampere/tests/test_cli.py`:
  - covered a two-step loop where the first score is rejected and written into attempt memory;
  - verified the second prompt sees the first rejected attempt;
  - verified the loop stops on acceptance, writes per-step records, and writes the loop summary;
  - covered the required `--attempts-dir` guard.
- Updated `/home/ubuntu/avo-ampere/README.md` and
  `/home/ubuntu/avo-ampere/knowledge/ampere.md` with the new command, stop conditions, and
  remaining limitation: this is a capped loop, not a full long-running supervisor.

Online research notes:

- Exa found Anthropic Agent SDK docs that separate caller-managed tool loops from managed agent
  loops and describe hooks that can validate, log, block, or transform lifecycle events. This
  supports keeping AVO's loop small and explicit around the existing validated one-step unit.
  Source: https://console.anthropic.com/docs/it/agent-sdk/stop-reasons
- Exa found Vercel AI SDK loop-control docs where `stopWhen` controls multi-step agent loops and
  the default maximum step count is 20 as a safety measure. This supports making a hard max-step
  cap first-class.
  Source: https://ai-sdk.dev/v7/docs/agents/loop-control
- Exa found a practical agent-loop writeup that treats max iterations as a key safety control and
  discusses loop fingerprinting/no-progress detection as a future defense. This matches the current
  tradeoff: max-step cap now, richer no-progress detection later.
  Source: https://github.com/stevekinney/stevekinney.net/blob/main/writing/agent-loops.md
- Exa found AgentFactory loop notes that call max iterations the primary safety net for autonomous
  loops.
  Source: https://agentfactory.panaversity.org/docs/General-Agents-Foundations/general-agents/ralph-wiggum-loop

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/cli.py tests/test_cli.py` passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_cli.py` passed, 9 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 91 tests.
- Whitespace:
  `git diff --check` in `/home/ubuntu/avo-ampere` passed.
- Live agent smoke:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage /tmp/avo-loop-lineage --knowledge knowledge/ampere.md --cwd /tmp/avo-loop-work --env-file ../avo/.env.local --attempts-dir /tmp/avo-loop-attempts --loop-json /tmp/avo-loop.json --max-steps 1 --timeout-s 300`
  passed against a temporary work copy. Anthropic returned `candidate_patch: ""`, selected the
  existing `candidates/cuda_mma_attention_seed.py` 32-token BF16 score, and the temporary lineage
  accepted one step with geomean `5.1666931286645514e-05` TFLOPS.

Tradeoffs and decision:

- The loop intentionally repeats the already-safe `evolve-once` unit instead of adding a separate
  supervisor state machine.
- Command failures and gate rejections do not stop the loop by themselves. They are recorded to
  attempts memory and can inform the next prompt until acceptance, cleanup failure, or max-step
  exhaustion.
- There is still no no-progress fingerprinting, dynamic stop policy, or full-session planner. Those
  belong after the one-step substrate has generated enough real attempt history to tune against.

## 2026-05-08 - Checkpoint 3.10: scored candidate source snapshots

Success criteria for this checkpoint:

- Bring accepted lineage commits closer to the architecture's `(source, score)` contract.
- Snapshot source for accepted candidate scores even when the agent chose no patch and only scored
  an existing candidate.
- Include companion CUDA/C++ source directories for seed modules such as
  `candidates/cuda_mma_attention_seed.py`.
- Keep rejected candidates score-only/no-commit and avoid broad runtime repository commits.
- Exclude generated artifacts such as `__pycache__`, `.pyc`, and compiled shared objects.

Implementation:

- Updated `/home/ubuntu/avo-ampere/avo/evolve.py`:
  - `finalize_attempt(..., source_root=...)` now snapshots source from the accepted
    `score_payload["candidate_path"]`, not only files touched by an accepted patch;
  - companion directories are inferred from scored Python seed modules, including the common
    `_seed.py` convention (`cuda_mma_attention_seed.py` -> `cuda_mma_attention/`);
  - source snapshots are limited to text source suffixes under `candidates/`;
  - symlink components, path traversal, `.git`, `__pycache__`, and paths outside `candidates/`
    are ignored;
  - accepted patched steps still store `patches/latest.patch`.
- Added `/home/ubuntu/avo-ampere/tests/test_evolve.py` coverage for a no-patch accepted candidate
  with a companion `.cpp/.cu` directory and generated artifacts that must be skipped.
- Updated `/home/ubuntu/avo-ampere/README.md` and
  `/home/ubuntu/avo-ampere/knowledge/ampere.md` to describe the stronger source artifact behavior
  and the remaining dependency-manifest limitation.

Online research notes:

- Exa found GEPA experiment-tracking docs that separate accepted candidate rows from rejected
  proposals and log structured optimization artifacts for inspection. This supports making accepted
  lineage artifacts more inspectable while keeping rejected attempts out of committed lineage.
  Source: https://github.com/gepa-ai/gepa/blob/main/docs/docs/guides/experiment-tracking.md
- Exa found W&B artifact-lineage guidance emphasizing that reproducibility needs exact code, not
  just metrics or a git hash when code may be uncommitted. This supports storing source snapshots in
  the lineage commit.
  Source: https://engineersofai.com/docs/ml/ml-with-python/weights-and-biases-integration
- Exa found Apache Hamilton lineage notes describing code versioning as lineage information tied to
  produced artifacts. This supports keeping source and score together in the accepted commit.
  Source: https://hamilton.incubator.apache.org/how-tos/use-hamilton-for-lineage/
- Exa found KAPSO's git-native experimentation paper, which represents attempts as isolated,
  reproducible git artifacts with code changes, evaluator configuration, logs, and outputs. This
  reinforces the decision to preserve accepted source artifacts directly in the lineage repository.
  Source: https://arxiv.org/pdf/2601.21526

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/evolve.py tests/test_evolve.py` passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_evolve.py tests/test_lineage.py` passed, 38 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 92 tests.
- Whitespace:
  `git diff --check` in `/home/ubuntu/avo-ampere` passed.

Tradeoffs and decision:

- The companion-directory inference intentionally follows the current seed naming convention rather
  than trying to parse arbitrary Python imports. It is enough for the CUDA seed layout in this repo
  and avoids a fragile dependency crawler.
- Direct `commit-score` remains score-only because it still has no source root. The stronger source
  behavior is attached to `finalize_attempt(...)`, which is what `evolve-once` and `evolve-loop`
  use.
- Accepted candidates that depend on files outside the scored module and companion directory still
  need an explicit dependency manifest later.

## 2026-05-08 - Checkpoint 3.11: attempt-history supervisor signals

Success criteria for this checkpoint:

- Add the smallest useful piece of stagnation/cycling recovery from `arch.md` without creating a
  second active agent process.
- Detect repeated unaccepted command/edit attempts in the existing attempts-dir memory.
- Detect a short exhaustion tail with no accepted candidates.
- Feed the signal into later agent prompts through attempt-history summaries.
- Preserve the hard loop cap, command allowlist, candidate patch validator, cleanup behavior, and
  lineage gate.

Implementation:

- Updated `/home/ubuntu/avo-ampere/avo/evolve.py`:
  - attempt-history summaries now append a supervisor signal when the last 3 unaccepted attempts
    share the same command/edit fingerprint;
  - attempt-history summaries now append an exhaustion signal when the last 5 attempts produced no
    accepted candidate;
  - fingerprints are deterministic SHA-256 prefixes over `next_command`, `candidate_patch`, and
    `files_to_inspect`;
  - the signal is text only: it asks the Anthropic planner to choose a materially different
    diagnostic or Ampere optimization direction, but it does not run tools or bypass gates.
- Added `/home/ubuntu/avo-ampere/tests/test_evolve.py` coverage for repeated unaccepted attempts
  and distinct unaccepted exhaustion tails.
- Updated `/home/ubuntu/avo-ampere/README.md` and
  `/home/ubuntu/avo-ampere/knowledge/ampere.md` to describe the supervisor signal and its limits.

Online research notes:

- Exa found a Claude-loop wrapper that uses circuit-breaker thresholds for no-progress detection,
  repeated errors, output decline, and repeated completion signals. This supports adding
  thresholded detection before building a richer supervisor.
  Source: https://github.com/DmitrySolana/ralph-claude-code/blob/main/CLAUDE.md
- Exa found a behavioral loop-detection skill that maintains a rolling window and escalates after
  repeated similar actions. This supports comparing recent attempts by normalized fingerprints.
  Source: https://github.com/oimiragieo/agent-studio/blob/main/.claude/skills/behavioral-loop-detection/SKILL.md
- Exa found Clawker loop-mode docs with circuit breakers for max loops, stagnation threshold,
  same-error threshold, per-iteration timeout, test-only loops, and completion signals. This matches
  AVO's current direction: hard cap first, lightweight stagnation signal second.
  Source: https://docs.clawker.dev/loops
- Exa found autonomous-debugging loop guidance that tracks attempted-solution hashes and repeated
  errors to avoid trying the same fix repeatedly. This supports hashing the agent's command/edit
  surface rather than relying only on natural-language summaries.
  Source: https://www.markaicode.com/fix-llm-infinite-loops-autonomous-debugging/

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/evolve.py tests/test_evolve.py` passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_evolve.py tests/test_cli.py` passed, 41 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 94 tests.
- Whitespace:
  `git diff --check` in `/home/ubuntu/avo-ampere` passed.

Tradeoffs and decision:

- This is not yet the full `arch.md` supervisor. It does not spawn a second Anthropic agent, inspect
  the full lineage deeply, or inject several generated strategy proposals.
- The signal is deliberately advisory and flows through prompt context. The existing evaluator and
  lineage gate remain the only acceptance authority.
- The fingerprint is intentionally coarse. It catches exact repeated command/edit surfaces and a
  short no-acceptance tail; it will not detect semantically equivalent rewrites with different patch
  text.

## 2026-05-08 - Checkpoint 3.12: benchmark provenance metadata

Success criteria for this checkpoint:

- Make accepted score payloads more auditable without changing the scoring function or gate.
- Record benchmark settings that affect timing: warmup, repeats, trials, seed, and case count.
- Record the intended A6000/sm86 target alongside observed non-secret environment metadata.
- Keep metadata collection safe in CPU-only unit tests and CUDA score runs.
- Verify the metadata on the actual A6000 path with a tiny GPU score smoke.

Implementation:

- Updated `/home/ubuntu/avo-ampere/avo/benchmark.py`:
  - added `benchmark_metadata(...)`;
  - every `score_backend(...)` summary now includes a top-level `benchmark` block;
  - metadata records `measured_at`, warmup/repeats/trials/seed/case count, target
    `NVIDIA RTX A6000` / `sm_86` / gencode, Python/platform, PyTorch version, PyTorch CUDA
    version, CUDA availability, and visible GPU properties when CUDA is available.
- Updated `/home/ubuntu/avo-ampere/tests/test_benchmark_math.py` and
  `/home/ubuntu/avo-ampere/tests/test_candidate_backend.py` for the new metadata block.
- Updated `/home/ubuntu/avo-ampere/README.md` and
  `/home/ubuntu/avo-ampere/knowledge/ampere.md` so humans and the agent know score payloads now
  carry timing provenance.

Online research notes:

- Exa found PyTorch benchmark utility guidance explaining that benchmark measurements are
  represented by serializable `Measurement` objects and grouped/interpreted by metadata. This
  supports storing settings and environment context in AVO score payloads, not only raw numbers.
  Source: https://github.com/pytorch/pytorch/blob/main/torch/utils/benchmark/README.md
- Exa found PyTorch benchmark docs emphasizing CUDA synchronization, warmups, replicates, and
  robust median computation. This matches AVO's current timing samples and motivates recording
  the timing settings that produced them.
  Source: https://docs.pytorch.org/docs/2.3/benchmark_utils.html
- Exa found TorchTitan benchmark submission guidance requesting hardware setup, actual performance
  reports with command/config details, dependency versions, and date/time. This supports adding
  target, environment, settings, and timestamp metadata to each score summary.
  Source: https://github.com/pytorch/torchtitan/blob/26181968/benchmarks/README.md
- Exa found PyTorch benchmark notes about benchmarking noise from interrupts, context switches,
  and clock frequency scaling. AVO does not eliminate those effects yet, but recording environment
  and replicate statistics keeps comparisons more inspectable.
  Source: https://github.com/pytorch/benchmark/blob/main/README.md

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/benchmark.py tests/test_benchmark_math.py tests/test_candidate_backend.py`
  passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_benchmark_math.py tests/test_candidate_backend.py`
  passed, 10 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 95 tests.
- Whitespace:
  `git diff --check` in `/home/ubuntu/avo-ampere` passed.
- A6000 GPU smoke:
  `uv run --extra cuda python -m avo score --backend torch-sdpa --seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16 --dtype bf16 --causal false --repeats 1 --warmup 1 --timeout-s 120`
  passed and the parsed payload reported `all_correct=true`, `sm=sm_86`,
  `gpu="NVIDIA RTX A6000"`, `torch_cuda="13.0"`, and `repeats=1`.

Tradeoffs and decision:

- The metadata is descriptive, not a benchmark-noise control mechanism. It does not pin clocks,
  record driver version, or enforce exclusive GPU access.
- The timestamp makes score payloads non-deterministic by design; lineage commits are already
  run-specific artifacts, and the audit value is worth it.
- This checkpoint improves reproducibility of future accepted score commits but does not itself
  improve kernel throughput.

## 2026-05-08 - Checkpoint 3.13: conservative FA2 build defaults

Success criteria for this checkpoint:

- Tighten FlashAttention-2 build defaults for the actual A6000 pod rather than relying on optimistic
  parallelism.
- Preserve the Ampere-only build target (`FLASH_ATTN_CUDA_ARCHS=80`).
- Keep caller-provided compile parallelism overrides possible.
- Update docs/knowledge so baseline setup is less likely to exhaust host memory.
- Record the failed/interrupted FA2 import preflight honestly.

Implementation:

- Attempted `uv run --extra baseline python -c 'import flash_attn; ...'` to check whether FA2 was
  importable. `uv` started building `flash-attn==2.8.3`, the build was interrupted during the
  session transition, no `flash-attn` process remained afterward, and `uv pip show flash-attn`
  confirmed it was not installed.
- Updated `/home/ubuntu/avo-ampere/avo/cli.py`:
  - `_baseline_build_env(...)` still forces `FLASH_ATTN_CUDA_ARCHS=80`;
  - it now defaults `MAX_JOBS=1` and `NVCC_THREADS=1` when the caller did not explicitly set them.
- Updated `/home/ubuntu/avo-ampere/tests/test_cli.py`:
  - covered the new conservative defaults;
  - covered preserving explicit `MAX_JOBS` and `NVCC_THREADS` overrides.
- Updated `/home/ubuntu/avo-ampere/README.md` and
  `/home/ubuntu/avo-ampere/knowledge/ampere.md` to use
  `FLASH_ATTN_CUDA_ARCHS=80 MAX_JOBS=1 NVCC_THREADS=1` for baseline installation on this 32 GB
  host.

Online research notes:

- Exa found a FlashAttention source-build guide that recommends `FLASH_ATTN_CUDA_ARCHS` to avoid
  building unnecessary architectures and `MAX_JOBS` to control host-memory pressure. This supports
  keeping the Ampere-only arch filter and explicitly limiting jobs.
  Source: https://vccv.cc/en/article/flash-attn-compile.html
- Exa found FlashAttention CUDA 13 installation issues where users report source builds for Python
  3.12/CUDA 13 and no prebuilt wheel path in some combinations. This explains why the import check
  started a source build instead of completing quickly.
  Source: https://github.com/Dao-AILab/flash-attention/issues/2064
- Exa found CUDA 13 / Torch 2.9+ FlashAttention install discussion showing wheel guessing can miss
  and fall back to source build. This reinforces treating FA2 setup as a controlled build step, not
  a cheap import check.
  Source: https://github.com/Dao-AILab/flash-attention/issues/2008
- Exa found FlashAttention build reports that `MAX_JOBS` does not fully bound memory because each
  NVCC job can spawn additional threads, and that `MAX_JOBS * NVCC_THREADS` drives worst-case
  memory use. This is the direct reason for defaulting both values to 1 on the 32 GB pod.
  Source: https://github.com/Dao-AILab/flash-attention/issues/2051

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/cli.py tests/test_cli.py` passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_cli.py` passed, 10 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 96 tests.
- Whitespace:
  `git diff --check` in `/home/ubuntu/avo-ampere` passed.
- Environment check:
  `uv pip show flash-attn` reported package not found after the interrupted build, so no baseline
  lineage was seeded in this checkpoint.

Tradeoffs and decision:

- `MAX_JOBS=1 NVCC_THREADS=1` will be slower, but it is safer for this 4-vCPU/32GB environment and
  better aligned with unattended baseline setup.
- We did not retry the full FA2 build immediately. The next baseline attempt should be an explicit,
  monitored install/seed step using the conservative flags, not an incidental import check.
- This checkpoint changes setup safety only; it does not yet produce an FA2 baseline score.

## 2026-05-08 - Checkpoint 3.14: baseline CUDA build preflight

Success criteria for this checkpoint:

- Diagnose the explicit FlashAttention-2 install failure after applying conservative build flags.
- Surface the relevant CUDA compatibility state through `avo env` without importing or building
  `flash_attn`.
- Prevent `seed-baseline` from entering the score worker when `flash_attn` is absent and PyTorch
  extension source builds are known to be blocked.
- Update the baseline setup docs/knowledge with the new preflight.

Implementation:

- Ran the explicit FA2 install command with the conservative settings:
  `FLASH_ATTN_CUDA_ARCHS=80 MAX_JOBS=1 NVCC_THREADS=1 uv pip install flash-attn==2.8.3 --no-build-isolation`.
  The build did not fail from memory pressure; it failed in PyTorch extension setup because the
  detected `nvcc` CUDA version was 12.9 while Torch was compiled with CUDA 13.0. The full log is in
  `/tmp/avo-flash-attn-install.log`.
- Verified environment facts:
  - `nvcc` resolves to `/usr/local/cuda-12.9/bin/nvcc`;
  - `nvcc --version` reports CUDA 12.9;
  - `uv run --extra cuda python -m avo env` reports Torch `2.11.0+cu130` with `torch_cuda` 13.0.
- Updated `/home/ubuntu/avo-ampere/avo/cli.py`:
  - added a `baseline_build` block to `avo env`;
  - mirrored PyTorch extension discovery by checking `CUDA_HOME`/`CUDA_PATH` first, then `PATH`;
  - parses `nvcc --version` and compares it with `torch.version.cuda`;
  - reports exact match, minor mismatch warning, major mismatch blocker, missing `nvcc`, or missing
    Torch CUDA;
  - keeps the existing FA2 build settings (`FLASH_ATTN_CUDA_ARCHS=80`, `MAX_JOBS=1`,
    `NVCC_THREADS=1`) in the status payload;
  - makes `seed-baseline` fail early only when `flash_attn` is not installed and the source-build
    environment is blocked.
- Updated `/home/ubuntu/avo-ampere/tests/test_cli.py` with parser, compatibility, env path,
  status, and early-rejection coverage.
- Updated `/home/ubuntu/avo-ampere/README.md` and
  `/home/ubuntu/avo-ampere/knowledge/ampere.md` to require checking `baseline_build` before an FA2
  install attempt.

Online research notes:

- Exa found PyTorch's primary `torch.utils.cpp_extension` source. It confirms that PyTorch finds
  CUDA through `CUDA_HOME`/`CUDA_PATH` or `nvcc`, parses `nvcc --version`, and raises the same CUDA
  mismatch error when the detected CUDA major version differs from `torch.version.cuda`.
  Source: https://github.com/pytorch/pytorch/blob/main/torch/utils/cpp_extension.py
- Exa also found a PyTorch issue discussing the same extension-build behavior and the practical
  workaround of pointing `CUDA_HOME` at the toolkit matching the Torch CUDA build.
  Source: https://github.com/pytorch/pytorch/issues/136845

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/cli.py tests/test_cli.py` passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_cli.py` passed, 15 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 101 tests.
- Whitespace:
  `git diff --check` passed in `/home/ubuntu/avo-ampere` and `/home/ubuntu/avo`.
- Environment check:
  `uv run --extra cuda python -m avo env` passed and reported `compatibility="major_mismatch"`,
  `flash_attn_installed=false`, `torch_cuda="13.0"`, `nvcc_cuda="12.9"`, and
  `ok_for_torch_extension_build=false`.

Tradeoffs and decision:

- This preflight cannot stop `uv run --extra baseline ...` from resolving the `baseline` extra
  before AVO code starts. The docs now use `uv pip install ...` after the preflight instead.
- A minor CUDA mismatch is treated as a warning rather than a blocker, matching PyTorch's extension
  behavior. The current environment is a major mismatch and remains blocked for FA2 source builds.
- No baseline lineage was seeded in this checkpoint; the next practical step is to provide a CUDA
  13.0 `nvcc` toolchain or use a Torch build compatible with the installed CUDA 12.9 toolkit.

## 2026-05-08 - Checkpoint 3.15: Python CUDA root selection for baseline builds

Success criteria for this checkpoint:

- Provide a matching CUDA 13 `nvcc` without changing the system CUDA 12.9 toolkit.
- Make the AVO baseline build environment choose the Python-installed CUDA root when it is
  compatible with Torch and the ambient CUDA root is not.
- Re-run the FA2 build far enough to verify it uses the CUDA 13 compiler, then record the remaining
  practical build-time blocker honestly.

Implementation:

- Installed Python CUDA compiler packages into the runtime venv:
  `nvidia-cuda-nvcc==13.0.88`, `nvidia-cuda-crt==13.0.88`, and `nvidia-nvvm==13.0.88`.
  This placed `nvcc`, `cuda.h`, and `libdevice.10.bc` under
  `/home/ubuntu/avo-ampere/.venv/lib/python3.12/site-packages/nvidia/cu13`.
- Updated `/home/ubuntu/avo-ampere/avo/cli.py`:
  - `_baseline_build_env(...)` now detects a single Python-installed `nvidia/cu*/bin/nvcc` root;
  - it checks that root's `nvcc` version against `torch.version.cuda`;
  - it overrides an incompatible ambient CUDA root, such as `/usr/local/cuda-12.9`, only when the
    Python CUDA root is compatible for PyTorch extension builds;
  - the `baseline_build.settings` payload now reports `CUDA_HOME` and `CUDA_PATH`.
- Updated `/home/ubuntu/avo-ampere/tests/test_cli.py` with coverage for Python CUDA root discovery,
  compatible-root selection, and preserving an already-compatible CUDA environment.
- Updated `/home/ubuntu/avo-ampere/README.md` and
  `/home/ubuntu/avo-ampere/knowledge/ampere.md` with the CUDA 13 nvcc wheel workaround.
- Retried FA2 installation with explicit `CUDA_HOME` from `_baseline_build_env(...)`:
  `FLASH_ATTN_CUDA_ARCHS=80 MAX_JOBS=1 NVCC_THREADS=1 uv pip install flash-attn==2.8.3 --no-build-isolation`.
  The build used the Python CUDA 13 `nvcc` and advanced into real CUDA compilation; it was stopped
  after completing only 15 of 73 Ninja objects because serial compilation was too slow for this
  checkpoint. `flash-attn` remained uninstalled. The partial log is in
  `/tmp/avo-flash-attn-install-cu13.log`.

Online research notes:

- Exa found the PyPI page for `nvidia-cuda-nvcc`, maintained by the NVIDIA CUDA Installer Team,
  showing CUDA 13.x nvcc wheels and dependencies on `nvidia-nvvm`, `nvidia-cuda-runtime`, and
  `nvidia-cuda-crt`.
  Source: https://pypi.org/project/nvidia-cuda-nvcc/
- Exa found NVIDIA forum discussion explaining that CUDA pip wheels may require additional
  environment setup for tool discovery. This supports AVO explicitly selecting the Python CUDA root
  instead of relying on the host `CUDA_HOME`.
  Source: https://forums.developer.nvidia.com/t/nvidia-cuda-nvcc-pip-wheel-installation/221307

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/cli.py tests/test_cli.py` passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_cli.py` passed, 18 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 104 tests.
- Whitespace:
  `git diff --check` passed in `/home/ubuntu/avo-ampere` and `/home/ubuntu/avo`.
- Environment check:
  `uv run --extra cuda python -m avo env` passed and reported `compatibility="exact"`,
  `torch_cuda="13.0"`, `nvcc_cuda="13.0"`, and `CUDA_HOME` under the venv's `nvidia/cu13` root.
- Package state:
  `uv pip show flash-attn` still reports package not found; `uv pip show nvidia-cuda-nvcc`
  reports version 13.0.88 installed.

Tradeoffs and decision:

- The Python CUDA root selection is limited to baseline build environments. It does not change the
  local candidate CUDA compiler path used by the custom kernel compile path.
- The full FA2 source build is now configuration-correct but operationally expensive. A later run
  can either let the serial build finish, raise parallelism after measuring peak memory, or seed a
  baseline from a compatible prebuilt wheel if one becomes available for this Python/Torch/CUDA
  tuple.
- No FA2 baseline lineage was seeded in this checkpoint.

## 2026-05-08 - Checkpoint 3.16: CUDA 13 candidate extension environment

Success criteria for this checkpoint:

- Stop candidate CUDA seeds from hard-coding `/usr/local/cuda-12.9`.
- Reuse the CUDA/Torch compatibility logic for both baseline builds and candidate
  `torch.utils.cpp_extension` builds.
- Prove a fresh CUDA candidate extension can build from an empty extension cache using the Python
  CUDA 13 toolchain while the host image still exports CUDA 12.9 paths.

Implementation:

- Added `/home/ubuntu/avo-ampere/avo/cuda_env.py` as the shared CUDA environment helper:
  - discovers a single Python-installed `nvidia/cu*` CUDA root;
  - checks `nvcc --version` against `torch.version.cuda`;
  - applies compatible Python CUDA roots to `CUDA_HOME`, `CUDA_PATH`, `CUDACXX`, `PATH`,
    `CPATH`, `LIBRARY_PATH`, and `LD_LIBRARY_PATH`;
  - filters ambient `/usr/local/cuda*` include/lib/bin entries from the worker process;
  - creates a local cached `libcudart.so` link shim when NVIDIA wheels only ship
    `libcudart.so.13`.
- Updated all CUDA candidate seed wrappers to call `prepare_torch_extension_env(...)` before
  importing `torch.utils.cpp_extension.load`. This matters because PyTorch computes its extension
  `CUDA_HOME` at import time.
- Moved the FlashAttention baseline CUDA preflight helper functions from `avo.cli` into the shared
  helper module and left the CLI as the reporting surface.
- Added reproducible CUDA 13 build dependencies to the `cuda` and `baseline` extras:
  `nvidia-cuda-nvcc==13.0.88`, `nvidia-cuda-crt==13.0.88`,
  `nvidia-nvvm==13.0.88`, and `nvidia-cuda-cccl==13.0.85`.
- Updated `/home/ubuntu/avo-ampere/README.md` so the baseline setup no longer depends on a manual
  `uv pip install` step for those CUDA wheels.

Discoveries:

- A cached identity extension initially hid the candidate seed problem, so a fresh
  `TORCH_EXTENSIONS_DIR` was required to validate the build path.
- Selecting the Python CUDA 13 `nvcc` was not enough by itself. The host shell exported
  `CPATH=/usr/local/cuda-12.9/include`, which caused CUDA 13 generated stubs to include CUDA 12.9
  CRT headers and fail with `__cudaLaunch` macro errors.
- After filtering `CPATH`, CUDA 13 compilation reached `cuda_fp16.h` and failed on missing
  `nv/target`. Exa confirmed this header is supplied by NVIDIA's CUDA CCCL package, specifically
  `nvidia-cuda-cccl` for the low-level header wheel.
- After installing CCCL, the link step failed on `-lcudart` because the CUDA 13 runtime wheel ships
  `libcudart.so.13` but no unversioned `libcudart.so`. The helper now creates a local symlink in
  `~/.cache/avo/cuda-links/cu13/lib` and exposes it through `LIBRARY_PATH` and `LD_LIBRARY_PATH`.

Online research notes:

- Exa found NVIDIA's CCCL setup docs, which recommend `cuda-cccl[cu13]` for CUDA 13 installs and
  explain that CCCL is distributed as Python packages.
  Source: https://nvidia.github.io/cccl/unstable/python/setup.html
- Exa found the PyPI page for `nvidia-cuda-cccl`, maintained by the NVIDIA CUDA Installer Team,
  with CUDA 13.0.85 and newer wheels.
  Source: https://pypi.org/project/nvidia-cuda-cccl/
- Exa found PyTorch CUDA-runtime linker discussions showing that PyTorch extension and custom
  builds still rely on `-lcudart`, which explains the need for an unversioned link target when
  using NVIDIA runtime wheels.
  Source: https://github.com/pytorch/pytorch/issues/163510

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/cuda_env.py tests/test_cli.py` passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_cli.py` passed, 20 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 106 tests.
- Whitespace:
  `git diff --check` passed in `/home/ubuntu/avo-ampere`.
- Environment check:
  `uv run --extra cuda python -m avo env` passed and still reports exact Torch/nvcc CUDA 13.0
  compatibility with the venv's `nvidia/cu13` root.
- Fresh CUDA candidate build:
  `TORCH_EXTENSIONS_DIR="$(mktemp -d /tmp/avo-ext-cu13.XXXXXX)" env -u CUDA_HOME -u CUDA_PATH
  uv run --extra cuda python -m avo score --backend candidate --candidate
  candidates/cuda_identity_seed.py --seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16
  --dtype bf16 --causal false --repeats 1 --warmup 1 --timeout-s 240` passed with
  `all_correct=true`.

Tradeoffs and decision:

- The cached `libcudart.so` link shim is local process infrastructure, not a tracked artifact. It is
  generated only when needed and avoids mutating installed wheel contents.
- Candidate extension builds are now CUDA 13 consistent. The first clean build still takes roughly
  one to two minutes on the 4-vCPU host, so agent-loop scoring should keep using small smoke shapes
  unless the extension cache is already warm.

## 2026-05-08 - Checkpoint 3.17: Accepted MMA seed lineage baseline

Success criteria for this checkpoint:

- Run the bounded agent loop through the now-working CUDA candidate extension path.
- Establish a first accepted lineage score for the Ampere BF16 MMA candidate before asking the
  agent to modify kernels.
- Confirm that accepted scored candidates snapshot source artifacts into the nested lineage.

Command:

`uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge
knowledge/ampere.md --attempts-dir benchmarks/attempts --loop-json benchmarks/latest-loop.json
--max-steps 1 --timeout-s 300 --env-file ../avo/.env.local`

Agent decision:

- The agent returned no candidate patch and chose to score the existing
  `candidates/cuda_mma_attention_seed.py` on a 32-token BF16 smoke test.
- The executed bounded command was:
  `avo score --backend candidate --candidate candidates/cuda_mma_attention_seed.py --seq-lens 32
  --total-tokens 32 --num-heads 1 --head-dim 16 --dtype bf16 --causal both --repeats 1 --warmup 1
  --timeout-s 300`

Result:

- Gate decision: accepted.
- Acceptance reason: candidate passed correctness and throughput gate.
- Previous best geomean: `0.0`.
- Candidate geomean: `6.23954365882066e-05` TFLOPS.
- Noncausal case: seq_len 32, BF16, head_dim 16, max_abs_error `0.001953125`,
  `0.6963520050048828` ms, `9.411332132165034e-05` TFLOPS.
- Causal case: seq_len 32, BF16, head_dim 16, max_abs_error `0.00390625`,
  `0.7921280264854431` ms, `4.136705040646883e-05` TFLOPS.
- Benchmark environment: Torch `2.11.0+cu130`, CUDA `13.0`, Python `3.12.13`,
  NVIDIA RTX A6000, `sm_86`.

Lineage artifacts:

- Nested lineage commit: `a52d8d7 evolve: accept candidate`.
- `scores/latest.json` was written.
- Accepted source snapshots were written under `sources/latest/`:
  - `candidates/cuda_mma_attention_seed.py`
  - `candidates/cuda_mma_attention/attention.cpp`
  - `candidates/cuda_mma_attention/attention_kernel.cu`

Tradeoffs and decision:

- This checkpoint intentionally accepted an unmodified seed because the lineage had no prior best.
  The useful output is a known-good source snapshot and a concrete baseline gate threshold.
- Future loop attempts now need to match or improve `6.23954365882066e-05` geomean TFLOPS on the
  same smoke gate, so regressions should be rejected and cleaned up by the patch cleanup path.

## 2026-05-08 - Checkpoint 3.18: Agent decision and compile guardrails

Success criteria for this checkpoint:

- Let the agent continue after the accepted MMA seed baseline and capture any bad-loop behavior.
- Convert repeated invalid autonomous decisions into validation feedback instead of wasted score
  or compile commands.
- Make `avo compile` useful for CUDA candidate sources on this CUDA 13/Torch 2.11 environment.

Rejected loop discoveries:

- The first post-baseline loop step tried to score the MMA seed at `head_dim=32` with no patch.
  The wrapper correctly rejected this because the current MMA seed accepts only sequence lengths
  16 or 32 with `head_dim=16`.
- The next loop step said it would extend/update the kernel and wrapper for `head_dim=32`, but
  still returned an empty `candidate_patch`. The orchestrator executed the command because the
  decision parser only checked patch syntax, not consistency between the proposed edit and patch.
- After adding a knowledge note, the agent tried `avo compile --candidate ...`, which is not a
  valid `avo compile` command.
- After command-argument validation was tightened, the agent tried
  `avo compile --source candidates/cuda_mma_attention_seed.py --out-dir build/mma_inspect`.
  This is the wrong source type because `avo compile` passes `--source` directly to `nvcc`.
- After source-path validation was tightened, the agent used the correct `.cu` source:
  `candidates/cuda_mma_attention/attention_kernel.cu`. That exposed a real compile-path gap:
  `avo compile` used ambient `/usr/local/cuda-12.9` and lacked PyTorch/Python include paths.

Runtime changes:

- Pushed runtime commit `2beabef docs: clarify mma smoke shape`, making the MMA smoke contract
  explicit in `knowledge/ampere.md`: seq_len 16/32, head_dim 16 only unless a candidate patch
  updates the wrapper and kernel first.
- Pushed runtime commit `28411ec fix: tighten agent decision boundaries`.
- Agent decision parsing now rejects an empty `candidate_patch` when `candidate_edit` describes a
  code change such as extending, updating, modifying, or fixing a candidate.
- `next_command` validation now checks basic subcommand arguments:
  - `avo compile` rejects `--candidate` and requires `--source` plus `--out-dir`;
  - compile sources must be repo-relative `.cu` files under `candidates/`;
  - candidate scores require `--backend candidate --candidate ...`;
  - candidate score paths must be repo-relative `.py` files under `candidates/`.
- `avo compile` now prepares the same CUDA 13 torch-extension environment used by candidate seeds,
  forwards PyTorch/Python/CUDA include directories, and adds `--expt-relaxed-constexpr` for parity
  with PyTorch extension compilation.
- The PyTorch include helper needed compatibility handling because Torch `2.11.0+cu130` exposes
  `torch.utils.cpp_extension.include_paths(device_type="cuda")`, not the older `cuda=True` form.

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/agent.py avo/cli.py avo/compile.py tests/test_agent.py
  tests/test_compile.py tests/test_cli.py` passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_agent.py tests/test_compile.py tests/test_cli.py` passed,
  55 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 114 tests.
- Whitespace:
  `git diff --check` passed in `/home/ubuntu/avo-ampere`.
- Manual compile parity check:
  `uv run --extra cuda python -m avo compile --source
  candidates/cuda_mma_attention/attention_kernel.cu --out-dir build/mma_inspect --timeout-s 180`
  passed. The emitted command used the venv CUDA 13 `nvcc`, the A6000 `sm_86` gencode,
  PyTorch/Python/CUDA include directories, and `--expt-relaxed-constexpr`.

Tradeoffs and decision:

- These changes do not make the MMA seed support `head_dim=32`; they prevent the loop from
  repeatedly describing that edit without providing a patch.
- `avo compile` is still compile-only and produces no score payload, so it will not enter lineage
  by itself. It is now a useful bounded diagnostic instead of an environment false negative.

## 2026-05-08 - Checkpoint 3.19: Accepted warp-row smoke baseline

Success criteria for this checkpoint:

- Continue the bounded loop after the decision guardrails and compile fixes.
- Use the supervisor signal from repeated unaccepted attempts to reset away from the MMA
  head-dimension expansion dead end.
- Establish a second accepted lineage candidate from a different CUDA seed.

Rejected discovery before acceptance:

- The first post-guardrail loop did reset strategy to `candidates/cuda_warp_rows_attention_seed.py`,
  but it tried production-like `seq_len=1024`, `head_dim=128`.
- The warp-row wrapper rejected that command with:
  `RuntimeError: cuda_warp_rows_attention_seed is a tiny multi-row correctness seed; got
  seq_len=1024, head_dim=128`.
- The source contract is `seq_len <= 128` and `head_dim <= 128`, so the runtime knowledge file
  was updated and pushed in commit `1a61443 docs: clarify warp-row smoke shape`.

Accepted command:

`uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge
knowledge/ampere.md --attempts-dir benchmarks/attempts --loop-json benchmarks/latest-loop.json
--max-steps 1 --timeout-s 300 --env-file ../avo/.env.local`

The agent selected:

`avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens
64 --total-tokens 64 --num-heads 1 --head-dim 32 --dtype bf16 --causal both --repeats 1 --warmup
1 --timeout-s 300`

Result:

- Gate decision: accepted.
- Acceptance reason: candidate passed correctness and throughput gate.
- Previous best geomean: `6.23954365882066e-05` TFLOPS.
- Candidate geomean: `0.0005428703821737525` TFLOPS.
- Noncausal case: seq_len 64, BF16, head_dim 32, max_abs_error `0.00390625`,
  `0.5724800229072571` ms, `0.0009158188565908014` TFLOPS.
- Causal case: seq_len 64, BF16, head_dim 32, max_abs_error `0.0078125`,
  `0.814624011516571` ms, `0.00032179753640206504` TFLOPS.
- Benchmark environment: Torch `2.11.0+cu130`, CUDA `13.0`, Python `3.12.13`,
  NVIDIA RTX A6000, `sm_86`.

Lineage artifacts:

- Nested lineage commit: `fe06c52 evolve: accept candidate`.
- `scores/latest.json` now reflects the warp-row score.
- Accepted source snapshots were written under `sources/latest/`:
  - `candidates/cuda_warp_rows_attention_seed.py`
  - `candidates/cuda_warp_rows_attention/attention.cpp`
  - `candidates/cuda_warp_rows_attention/attention_kernel.cu`

Tradeoffs and decision:

- This accepted candidate is not a code improvement; it is a better existing seed baseline on a
  valid smoke shape.
- The current lineage best now represents the warp-row seed at `seq_len=64`, `head_dim=32`. Future
  accepted steps must match or improve `0.0005428703821737525` geomean TFLOPS while preserving
  correctness.

## 2026-05-08 - Checkpoint 3.20: Warp-row shared-staging scale-up and agent field guard

Success criteria for this checkpoint:

- Add one fresh Ampere-specific research note to the runtime knowledge base before the next loop.
- Continue the bounded loop against the current warp-row lineage best.
- Capture and fix any new agent structured-output reliability issue.

Online research:

- Exa found NVIDIA CUTLASS's CuTeDSL Ampere FlashAttention v2 example. The example targets Ampere
  FlashAttention-style forward attention and combines 128-bit `cp.async` Q/K/V global-to-shared
  copies, Ampere BF16/FP16 tensor-core MMA via `MmaF16BF16Op(..., (16, 8, 16))`, register
  pipelining for shared-to-register copies, online softmax with output rescaling, and head-dim
  padding to multiples of 32. It uses default 128x128 m/n tiles with 128 threads, but the local
  candidate search should still stay on small smoke shapes until correctness is stable.
  Source: https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py
- Runtime commit `610c72a docs: add cutlass ampere fa2 note` added this concise direction note to
  `knowledge/ampere.md`.

Accepted command:

`uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge
knowledge/ampere.md --attempts-dir benchmarks/attempts --loop-json benchmarks/latest-loop.json
--max-steps 1 --timeout-s 300 --env-file ../avo/.env.local`

The agent selected:

`avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens
128 --total-tokens 128 --num-heads 1 --head-dim 64 --dtype bf16 --causal both --repeats 1
--warmup 1 --trials 1`

Result:

- Gate decision: accepted.
- Acceptance reason: candidate passed correctness and throughput gate.
- Previous best geomean: `0.0005428703821737525` TFLOPS.
- Candidate geomean: `0.003413329681044312` TFLOPS.
- Noncausal case: seq_len 128, BF16, head_dim 64, max_abs_error `0.001953125`,
  `0.8708800077438354` ms, `0.00481616751183216` TFLOPS.
- Causal case: seq_len 128, BF16, head_dim 64, max_abs_error `0.0078125`,
  `0.8669120073318481` ms, `0.0024191059556950213` TFLOPS.
- Benchmark environment: Torch `2.11.0+cu130`, CUDA `13.0`, Python `3.12.13`,
  NVIDIA RTX A6000, `sm_86`.

Lineage artifacts:

- Nested lineage commit: `f360509 evolve: accept candidate`.
- Only `scores/latest.json` changed because the accepted source snapshot was already the warp-row
  seed from the previous accepted step.

Agent reliability discovery and fix:

- The accepted decision had a malformed-looking `expected_effect` value containing
  `<parameter name="risk">...`, while the actual `risk` field was also present and correct.
- This did not affect command execution or scoring, but it is a structured-output contamination
  pattern that should be retried instead of recorded.
- Runtime commit `6eb3dca fix: reject contaminated agent fields` now rejects ordinary string fields
  containing tool parameter markup. `candidate_patch` remains governed by the raw-diff validator.

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/agent.py tests/test_agent.py` passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_agent.py` passed, 34 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 115 tests.
- Whitespace:
  `git diff --check` passed in `/home/ubuntu/avo-ampere`.

Tradeoffs and decision:

- This is still a no-patch scale-up of the existing warp-row seed, not an optimization. It gives the
  loop a stronger correctness/performance checkpoint before asking for code edits.
- The current lineage best is now `0.003413329681044312` geomean TFLOPS on `seq_len=128`,
  `head_dim=64`, which future accepted candidates must match or improve.

## 2026-05-08 - Checkpoint 3.21: Warp-row head_dim 128 smoke accept

Success criteria for this checkpoint:

- Run one bounded evolution step against the current `seq_len=128`, `head_dim=64` lineage best.
- Keep any result limited to the lineage artifact unless the agent proposes and passes a code patch.
- Record whether the existing warp-row seed can handle `head_dim=128` at the smoke cap.

Accepted command:

`uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge
knowledge/ampere.md --attempts-dir benchmarks/attempts --loop-json benchmarks/latest-loop.json
--max-steps 1 --timeout-s 300 --env-file ../avo/.env.local`

The agent selected:

`avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens
128 --total-tokens 128 --num-heads 1 --head-dim 128 --dtype bf16 --causal both --repeats 1
--warmup 1 --trials 1 --timeout-s 300`

Result:

- Gate decision: accepted.
- Acceptance reason: candidate passed correctness and throughput gate.
- Previous best geomean: `0.003413329681044312` TFLOPS.
- Candidate geomean: `0.007001040892301204` TFLOPS.
- Noncausal case: seq_len 128, BF16, head_dim 128, max_abs_error `0.00390625`,
  `0.9043840169906616` ms, `0.0092754934213821` TFLOPS.
- Causal case: seq_len 128, BF16, head_dim 128, max_abs_error `0.0078125`,
  `0.7937279939651489` ms, `0.005284309022599703` TFLOPS.
- Benchmark environment: Torch `2.11.0+cu130`, CUDA `13.0`, Python `3.12.13`,
  NVIDIA RTX A6000, `sm_86`.

Lineage artifacts:

- Nested lineage commit: `1ed290a evolve: accept candidate`.
- Only `scores/latest.json` changed in the nested lineage repo.
- The accepted source remains the same warp-row seed snapshot; no outer runtime repo files changed.

Verification:

- `benchmarks/latest-loop.json` reports `accepted: true`, `completed_steps: 1`, and
  `stopped_reason: accepted`.
- `git diff --check` passed in `/home/ubuntu/avo-ampere`.
- Outer runtime repo status was clean after the run.
- Nested lineage status was clean after commit `1ed290a`.

Tradeoffs and decision:

- This is another no-patch scale-up, but it validates the current warp-row seed at the wrapper's
  head-dim smoke cap before asking the agent for larger structural kernel edits.
- The current lineage best is now `0.007001040892301204` geomean TFLOPS on `seq_len=128`,
  `head_dim=128`, which future accepted candidates must match or improve.

## 2026-05-08 - Checkpoint 3.22: Reject out-of-cap shape-only scores

Success criteria for this checkpoint:

- Add one fresh Ampere/FA2 implementation guardrail to the runtime knowledge base.
- Run one bounded loop step from the current `seq_len=128`, `head_dim=128` lineage best.
- If the agent repeats a known shape-boundary mistake, turn it into a pre-execution validation
  guard instead of letting future loops waste score calls.

Online research:

- Exa found Dao-AILab's CuTe FlashAttention forward implementation, which reimplements the
  FlashAttention forward kernel path in CuTe and references both the SM80 and SM90 kernel lineage.
  Useful Ampere guardrails for local patches: FP16/BF16 only, Q/K/V head dimensions aligned to
  multiples of 8 for 16-byte access, tile-N divisible by 16, thread count a multiple of 32, and
  shared-memory budgeting as Q plus staged K/V tiles.
  Source: https://github.com/Dao-AILab/flash-attention/blob/58fe37fb/flash_attn/cute/flash_fwd.py
- Runtime commit `52e990e docs: add dao cute fa2 guardrails` added this to
  `knowledge/ampere.md` and nudged future steps toward bounded code edits rather than another
  shape-only warp-row score.

Rejected command:

`uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge
knowledge/ampere.md --attempts-dir benchmarks/attempts --loop-json benchmarks/latest-loop.json
--max-steps 1 --timeout-s 300 --env-file ../avo/.env.local`

The agent selected:

`avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens
256 --total-tokens 256 --num-heads 1 --head-dim 128 --dtype bf16 --causal both --repeats 1
--warmup 1 --trials 1 --timeout-s 300`

Result:

- Gate decision: rejected.
- Rejection reason: candidate failed correctness.
- Previous best geomean: `0.007001040892301204` TFLOPS.
- Candidate geomean: `0.0` TFLOPS.
- Both causal and noncausal cases failed before timing with:
  `RuntimeError: cuda_warp_rows_attention_seed is a tiny multi-row correctness seed; got
  seq_len=256, head_dim=128`.
- No candidate patch was proposed, and no lineage commit was created.

Agent reliability fix:

- Runtime commit `f6eaf1c fix: reject unpatched seed cap scores` now rejects this class of
  decision before execution.
- The validator knows the unpatched local seed caps:
  `cuda_warp_rows_attention_seed.py` supports `seq_len <= 128` and `head_dim <= 128`;
  `cuda_mma_attention_seed.py` supports `seq_len` 16 or 32 and `head_dim` 16.
- Larger scores remain allowed when the decision includes a non-empty raw candidate patch, because
  that patch may update the wrapper/kernel to support the new shape.
- `build_repo_context` also tells the agent these caps directly.

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/agent.py tests/test_agent.py` passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_agent.py` passed, 37 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 118 tests.
- Whitespace:
  `git diff --check` passed in `/home/ubuntu/avo-ampere`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `f6eaf1c479f1ee7363d5fb9fa401ca02ad1af150`.

Tradeoffs and decision:

- This was not a kernel optimization, but it removes a now-observed wasted step from the agent
  loop and keeps future shape increases tied to explicit candidate patches.
- The current lineage best remains `0.007001040892301204` geomean TFLOPS on `seq_len=128`,
  `head_dim=128`.

## 2026-05-08 - Checkpoint 3.23: Structured planning failure and multi-head smoke accept

Success criteria for this checkpoint:

- Verify the seed-cap validator against the live agent loop.
- If invalid planning still escapes the retry loop as a traceback, convert it into structured loop
  output.
- Continue only one bounded step after that fix and record whether the result is a code patch or
  another shape-only score.

Live validator result:

- Re-running the loop after `f6eaf1c` correctly rejected the repeated no-patch `seq_len=256`
  warp-row score during decision validation.
- The model repeated the invalid shape-only plan through all three planning attempts, so the CLI
  exited with a raw `ValueError` traceback instead of writing a structured `latest-loop.json`.
- Runtime commit `ed241e6 fix: record invalid planning attempts` now catches planning validation
  `ValueError`s inside evolve steps and converts them into a synthetic failed attempt with
  `stderr_tail` explaining the planning validation failure. The loop still exits nonzero when no
  candidate is accepted, but the failure is now recorded in the usual loop/attempt JSON shape.

Verification for the planning-failure fix:

- Focused lint:
  `uv run --extra dev ruff check avo/cli.py avo/evolve.py tests/test_cli.py` passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_cli.py tests/test_evolve.py` passed, 53 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 119 tests.
- Whitespace:
  `git diff --check` passed in `/home/ubuntu/avo-ampere`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `ed241e68bc79fc7aa28dfea23a3e7a66f2ca5170`.

Accepted command after the fix:

`uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge
knowledge/ampere.md --attempts-dir benchmarks/attempts --loop-json benchmarks/latest-loop.json
--max-steps 1 --timeout-s 300 --env-file ../avo/.env.local`

The agent selected:

`avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens
128 --total-tokens 512 --num-heads 4 --head-dim 128 --dtype bf16 --causal both --repeats 1
--warmup 1 --timeout-s 300`

Result:

- Gate decision: accepted.
- Acceptance reason: candidate passed correctness and throughput gate.
- Previous best geomean: `0.007001040892301204` TFLOPS.
- Candidate geomean: `0.10830947571120902` TFLOPS.
- Noncausal case: batch 4, heads 4, seq_len 128, BF16, head_dim 128, max_abs_error
  `0.00390625`, `1.0499839782714844` ms, `0.12782835812500043` TFLOPS.
- Causal case: batch 4, heads 4, seq_len 128, BF16, head_dim 128, max_abs_error
  `0.015625`, `0.7312639951705933` ms, `0.09177104909198282` TFLOPS.
- Benchmark environment: Torch `2.11.0+cu130`, CUDA `13.0`, Python `3.12.13`,
  NVIDIA RTX A6000, `sm_86`.

Lineage artifacts:

- Nested lineage commit: `07f1441 evolve: accept candidate`.
- Only `scores/latest.json` changed in the nested lineage repo.
- The accepted source remains the same warp-row seed snapshot; no outer runtime repo files changed.

Tradeoffs and decision:

- This validates multi-head/batch indexing for the warp-row seed at the current smoke caps, which is
  useful before larger wrapper/kernel edits.
- It is still not a kernel optimization. The large TFLOPS jump mainly comes from changing the
  workload from single-head, 128 total tokens to four heads and 512 total tokens, so future steps
  should not treat this as evidence of a faster kernel.
- The current lineage best is now `0.10830947571120902` geomean TFLOPS on the multi-head
  `seq_len=128`, `head_dim=128`, `total_tokens=512`, `num_heads=4` smoke.

## 2026-05-08 - Checkpoint 3.24: Cap unpatched workload scaling

Success criteria for this checkpoint:

- Prevent the agent from getting more accepted lineage commits by increasing only batch/head
  parallelism on unchanged seed code.
- Keep larger workload scores available when the agent provides a candidate patch that updates the
  wrapper or kernel.

Reliability fix:

- Runtime commit `95340e4 fix: cap unpatched seed workload scores` extends the seed-score
  validator beyond `seq_len` and `head_dim`.
- For unpatched `cuda_warp_rows_attention_seed.py` scores, the validator now caps smoke commands at
  `seq_len <= 128`, `head_dim <= 128`, `total_tokens <= 512`, and `num_heads <= 4`.
- For unpatched `cuda_mma_attention_seed.py` scores, the validator now caps smoke commands at
  `seq_len` 16 or 32, `head_dim` 16, `total_tokens <= 32`, and `num_heads = 1`.
- Larger scores are still allowed with a non-empty raw `candidate_patch`, so genuine wrapper/kernel
  extensions are not blocked.
- `build_repo_context` now tells the agent these workload caps directly.

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/agent.py tests/test_agent.py` passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_agent.py` passed, 40 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 122 tests.
- Whitespace:
  `git diff --check` passed in `/home/ubuntu/avo-ampere`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `95340e449adc6188d91631707b3d62a52fc3db29`.

Tradeoffs and decision:

- This is a guardrail, not an optimization. It keeps the latest multi-head smoke baseline available
  while reducing the chance that future no-patch decisions are accepted only because the workload
  exposes more parallelism.
- The current lineage best remains `0.10830947571120902` geomean TFLOPS on the multi-head
  warp-row smoke from nested lineage commit `07f1441`.

## 2026-05-08 - Checkpoint 3.25: Reject env as source inspection

Success criteria for this checkpoint:

- Run one bounded loop after the workload cap guard.
- If the agent uses `avo env` as a substitute for source-file inspection, reject that pattern at
  planning time.

Loop result before the fix:

- The agent stayed within the tighter workload caps but selected `avo env` while describing an
  intent to inspect `candidates/cuda_warp_rows_attention_seed.py`.
- The command succeeded and reported the CUDA/build environment, but there was no score payload and
  no candidate patch.
- Gate decision: no gate decision; the loop stopped at `max_steps`.
- No lineage commit was created; nested lineage remained at `07f1441`.
- The decision also incorrectly described the `total_tokens=512` cap as a wrapper cap. It is an
  orchestrator planning guard, not a source-level wrapper limit.

Reliability fix:

- Runtime commit `5f69266 fix: reject env as source inspection` rejects `avo env` unless the
  planning text is about CUDA, Torch, NVCC, FlashAttention installation, or build/environment
  diagnostics.
- The prompt and repo context now state that `avo env` is only for CUDA/build diagnostics, not
  source-file inspection.
- `avo env` remains allowed for real environment checks.

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/agent.py tests/test_agent.py` passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_agent.py` passed, 42 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 124 tests.
- Whitespace:
  `git diff --check` passed in `/home/ubuntu/avo-ampere`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `5f692664c20c46720e1218faf2c18d419b998a49`.

Tradeoffs and decision:

- This removes another low-value planning escape hatch. The agent must now choose a real bounded
  score/compile command or provide a candidate patch instead of spending a step on an environment
  command when it wants file context.
- The current lineage best remains `0.10830947571120902` geomean TFLOPS on nested lineage commit
  `07f1441`.

## 2026-05-08 - Checkpoint 3.26: MMA seed baseline remains noncompetitive

Success criteria for this checkpoint:

- Run one bounded loop after the `avo env` source-inspection guard.
- Confirm whether the agent now chooses a real score/compile step or patch under the latest
  planning rules.

Loop result:

- The agent selected a valid no-patch score command for `candidates/cuda_mma_attention_seed.py`:
  `seq_len=32`, `head_dim=16`, `total_tokens=32`, `num_heads=1`, BF16, both causal modes.
- The command passed correctness for both cases.
- Noncausal case: max_abs_error `0.001953125`, `0.7282559871673584` ms,
  `8.999033465541473e-05` TFLOPS.
- Causal case: max_abs_error `0.00390625`, `0.6895359754562378` ms,
  `4.7521813460593916e-05` TFLOPS.
- Candidate geomean: `6.539498372773744e-05` TFLOPS.
- Gate decision: rejected, because the current lineage best is `0.10830947571120902` geomean
  TFLOPS from nested lineage commit `07f1441`.
- No candidate patch was applied, no lineage commit was created, and nested lineage remained at
  `07f1441`.

Tradeoffs and decision:

- This is useful as a negative result: the MMA seed is correct at its maximum unpatched smoke shape,
  but it is still only a correctness foothold.
- Future MMA work should bring an actual candidate patch that expands the kernel/wrapper path or
  improves tiling, not another no-patch score of the same supported shape.

## 2026-05-08 - Checkpoint 3.27: Reject compile as source inspection

Success criteria for this checkpoint:

- Run one bounded loop after recording the MMA seed negative result in runtime knowledge.
- If the agent uses `avo compile` as a substitute for source-file inspection, reject that pattern
  at planning time while preserving legitimate compile diagnostics.

Loop result before the fix:

- The agent avoided the rejected MMA baseline but selected:
  `avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu --out-dir build/warp_rows_inspect`.
- The planning text said the goal was to inspect boundary handling, indexing logic, loop bounds,
  tile edge conditions, and causal masking for a suspected `seq_len=256` correctness issue.
- The compile command succeeded, but it produced only an object file and compiler command payload.
  It did not provide source inspection, correctness data, performance data, or a candidate patch.
- No lineage commit was created; nested lineage remained at `07f1441`.

Reliability fix:

- Runtime commit `d306aa6 fix: reject compile as source inspection` rejects no-patch `avo compile`
  decisions unless the planning text is about CUDA build/compilation diagnostics.
- `avo compile` remains allowed for actual build diagnostics and for build-checking a non-empty
  `candidate_patch`.
- The prompt and repo context now state that `avo compile` is not a source-inspection command.

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/agent.py tests/test_agent.py` passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_agent.py` passed, 44 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 126 tests.
- Whitespace:
  `git diff --check` passed in `/home/ubuntu/avo-ampere`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `d306aa6f0b61362579b1eb4ec875d65a57a83069`.

Tradeoffs and decision:

- This is another planning guardrail, not a kernel optimization. It keeps `avo compile` available
  for cases where it can actually answer the proposed question.
- Future loops must now either provide a candidate patch, run a bounded candidate score, or use
  compile for a real build/compilation diagnostic.

## 2026-05-08 - Checkpoint 3.28: Guarded planning failure is structured

Success criteria for this checkpoint:

- Run one bounded loop after the compile-as-inspection guard.
- Confirm that invalid edit-without-patch plans are captured as structured attempts rather than
  leaking as raw tracebacks or workspace changes.

Loop result:

- The agent produced invalid decisions for all three planning attempts: its `candidate_edit`
  described a code change but `candidate_patch` was empty.
- The loop recorded a synthetic failed `agent-plan` attempt with:
  `candidate_patch must be non-empty when candidate_edit describes a code change`.
- No bounded command was executed, no candidate patch was applied, and no lineage commit was created.
- Runtime repo, docs repo, and nested lineage were clean after the run; lineage remained at
  `07f1441`.

Tradeoffs and decision:

- This is the desired failure mode after the stricter guardrails: invalid plans consume a bounded
  attempt and become history for the next prompt instead of mutating the workspace.
- The next useful step is to see whether the attempt history now pushes the agent into either a
  real candidate patch or a bounded score that can affect lineage.

## 2026-05-08 - Checkpoint 3.29: Compile diagnostics now emit ptxas resources

Success criteria for this checkpoint:

- Run one bounded loop after the structured planning failure.
- If the agent chooses a legitimate compile diagnostic, make sure the command produces the
  resource data the agent expects.

Loop result before the fix:

- The agent selected a no-patch compile diagnostic for
  `candidates/cuda_warp_rows_attention/attention_kernel.cu`.
- The decision expected ptxas resource statistics for register usage and shared memory before
  adding WMMA complexity.
- The compile succeeded, but `stderr_tail` was empty because `avo compile` did not request ptxas
  verbosity.
- No lineage commit was created; nested lineage remained at `07f1441`.

Runtime improvements:

- Runtime commit `eeaa4d8 feat: include ptxas stats in compile output` adds
  `--ptxas-options=-v` to `avo compile`.
- A real compile of the warp-row kernel now reports:
  - BF16 entry point: 48 registers, 1 barrier, 16896 bytes shared memory, no spills.
  - Half entry point: 48 registers, 1 barrier, 16896 bytes shared memory, no spills.
  - FP32 entry point: 56 registers, 1 barrier, 33280 bytes shared memory, no spills.
- Runtime commit `6c70869 docs: record warp row ptxas baseline` stores those resource numbers in
  `knowledge/ampere.md` for future agent prompts.

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/compile.py tests/test_compile.py` passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_compile.py` passed, 2 tests.
- Real compile:
  `uv run --extra cuda python -m avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu --out-dir build/ptxas_check`
  passed and returned ptxas resource output.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 126 tests.
- Whitespace:
  `git diff --check` passed in `/home/ubuntu/avo-ampere`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `6c70869cd2173a03b2ede7fec5af5a4bbbd73a6f`.

Tradeoffs and decision:

- This keeps compile diagnostics useful without broadening the command allowlist.
- The ptxas numbers are not a candidate improvement by themselves, but they give the next kernel
  patch a concrete register/shared-memory baseline.

## 2026-05-08 - Checkpoint 3.30: Tiled seed ptxas baseline

Success criteria for this checkpoint:

- Run one bounded loop after recording warp-row ptxas resources.
- If the agent pivots to a new local seed, capture any useful compile diagnostics in runtime
  knowledge.

Loop result:

- The agent pivoted from warp-row/MMA to the tiled seed and compiled:
  `candidates/cuda_tiled_attention/attention_kernel.cu`.
- The compile succeeded and emitted ptxas resource data through the new diagnostic path.
- BF16 entry point: 40 registers, 1 barrier, no spills.
- Half entry point: 40 registers, 1 barrier, no spills.
- FP32 entry point: 40 registers, 1 barrier, no spills.
- Double entry point: 48 registers, 1 barrier, no spills.
- ptxas did not report static shared-memory allocation for these entry points.
- No score was run and no lineage commit was created; nested lineage remained at `07f1441`.

Runtime update:

- Runtime commit `30883e5 docs: record tiled ptxas baseline` stores the tiled seed resource
  baseline in `knowledge/ampere.md`.

Tradeoffs and decision:

- The tiled compile result is useful context but does not prove correctness or performance.
- Future tiled work should run a bounded candidate score or provide a real candidate patch rather
  than repeating a no-patch compile diagnostic.

## 2026-05-08 - Checkpoint 3.31: Normalize repeated compile attempts

Success criteria for this checkpoint:

- Run one bounded loop after recording the tiled ptxas baseline.
- If the agent repeats the same compile diagnostic with only a different `--out-dir`, make that
  repeat visible to the supervisor.

Loop result before the fix:

- The agent repeated the tiled seed compile under a new output directory:
  `build/tiled_baseline_verify` instead of `build/tiled_baseline_check`.
- The compile succeeded and produced the same ptxas pattern: BF16/Half/FP32 entry points at
  40 registers and 1 barrier with no spills, and the double entry point at 48 registers.
- No score was run and no candidate patch was applied, so no lineage commit was created.
- The existing repeated-attempt fingerprint included the full `next_command`, so changing
  `--out-dir` made equivalent compile diagnostics look different.

Reliability fix:

- Runtime commit `74310e1 fix: normalize repeated compile attempts` fingerprints `avo compile`
  attempts by `--source` instead of the full command string.
- Repeating the same compile source with different throwaway build directories now triggers the
  repeated-unaccepted supervisor signal.

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/evolve.py tests/test_evolve.py` passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_evolve.py` passed, 33 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 127 tests.
- Whitespace:
  `git diff --check` passed in `/home/ubuntu/avo-ampere`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `74310e1db24321237734f91e0e0062b1cc62be77`.

Tradeoffs and decision:

- This does not reject compile diagnostics outright. It makes repeated no-lineage compile work
  more visible in attempt history, so the next prompt can push toward scoring or patching.

## 2026-05-08 - Checkpoint 3.32: Cap unpatched tiled scores to validated tiny smoke

Success criteria for this checkpoint:

- Do a small online research refresh for Ampere FlashAttention structure before the next local
  change.
- Score the tiled seed instead of repeating compile diagnostics.
- If the tiled seed has a narrower correctness envelope than its wrapper allows, encode that in
  planning validation.

Research refresh:

- Exa returned the same primary-source direction already captured in `knowledge/ampere.md`:
  NVIDIA CUTLASS CuTeDSL Ampere FA2 uses cp.async global-to-shared copies, Ampere tensor-core MMA,
  register pipelining, and online softmax; Dao-AILab's CuTe/SM80 path has the same broad structure.
- This reinforces that tiled and warp-row seeds are structural footholds only; moving toward FA2
  requires real patches, not more compile-only diagnostics.

Local scores:

- Larger tiled smoke:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_tiled_attention_seed.py --seq-lens 128 --total-tokens 512 --num-heads 4 --head-dim 128 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  failed correctness.
- Larger tiled noncausal case: max_abs_error `0.485504150390625`, `1.4002879858016968` ms,
  TFLOPS `0.0`.
- Larger tiled causal case: max_abs_error `1.4482421875`, `1.358016014099121` ms,
  TFLOPS `0.0`.
- Tiny tiled smoke:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_tiled_attention_seed.py --seq-lens 16 --total-tokens 16 --num-heads 1 --head-dim 16 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
  passed correctness.
- Tiny tiled noncausal case: max_abs_error `0.00390625`, `1.0389440059661865` ms,
  `1.576985853512228e-05` TFLOPS.
- Tiny tiled causal case: max_abs_error `0.015625`, `0.8979840278625488` ms,
  `9.122656690786842e-06` TFLOPS.
- Tiny tiled geomean: `1.1994290536675978e-05` TFLOPS.

Reliability fix:

- Runtime commit `f478eae fix: cap unpatched tiled seed scores` caps no-patch
  `cuda_tiled_attention_seed.py` scores to the validated tiny smoke:
  `seq_len=16`, `head_dim=16`, `total_tokens<=16`, and `num_heads=1`.
- Larger tiled scores now require a non-empty `candidate_patch` that fixes or extends the kernel.
- Runtime knowledge now records both the tiny passing score and the larger failing score.

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/agent.py tests/test_agent.py` passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_agent.py` passed, 47 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 130 tests.
- Whitespace:
  `git diff --check` passed in `/home/ubuntu/avo-ampere`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `f478eae4748d24cc207dbc0dc621f7e46ffd5475`.

Tradeoffs and decision:

- The tiled seed remains useful as a simple correctness reference and possible patch target, but
  it is not a viable no-patch branch for the current 128-token smoke workload.
- The current lineage best remains `0.10830947571120902` geomean TFLOPS on nested lineage commit
  `07f1441`.

## 2026-05-08 - Checkpoint 3.33: Clarify invalid patch retry feedback

Success criteria for this checkpoint:

- Run one bounded loop after capping unpatched tiled seed scores.
- If the agent again describes a code change without supplying a raw patch, make the retry feedback
  explicit enough to distinguish patching from no-edit diagnostics.

Loop result before the fix:

- The loop failed during decision validation before executing a command or creating a lineage
  commit.
- Validation rejected the decision because `candidate_edit` described a code change while
  `candidate_patch` was empty.
- The previous retry prompt reported the validation error, but did not spell out the exact raw
  unified-diff requirement or the valid no-edit alternative.

Reliability fix:

- Runtime commit `bcb7b74 fix: clarify invalid patch retry feedback` adds targeted retry guidance
  for this validation error.
- If the next step changes code, the retry prompt now says `candidate_patch` must be a raw unified
  diff starting with `diff --git`.
- If no code changes are needed, it tells the agent to make `candidate_edit` say `No edit; ...`
  and describe only the bounded score, compile, or environment diagnostic.

Verification:

- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 131 tests.
- Whitespace:
  `git diff --check` passed in `/home/ubuntu/avo-ampere`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `bcb7b7465fb96cda88200718afd2d5a1fbc14408`.

Tradeoffs and decision:

- This does not loosen validation. It keeps the raw-patch gate intact and makes the correction path
  more concrete for Anthropic retries.
- The current lineage best remains `0.10830947571120902` geomean TFLOPS on nested lineage commit
  `07f1441`.

## 2026-05-08 - Checkpoint 3.34: Harden patch planning and record naive baseline

Success criteria for this checkpoint:

- Continue bounded loop execution after the first invalid-patch retry fix.
- Turn any remaining planning or patch-substrate failures into explicit runtime guardrails.
- Record any new no-patch baseline scores that should not be repeated.

Research refresh:

- Exa primary-source refresh again pointed at the Ampere FA2 structure we already want:
  NVIDIA CUTLASS CuTeDSL Ampere FA2 uses `cp.async`, tensor-core MMA, register pipelining,
  and online softmax; Dao-AILab's SM80 forward path uses the same broad pattern with explicit
  async-copy staging and tiled MMA.
- This reinforces that useful candidate edits should move the warp-row or MMA footholds toward
  those mechanisms, not keep scoring slow scalar references.

Loop results and reliability fixes:

- After runtime commit `bcb7b74`, the next loop still failed planning validation after three
  retries with `candidate_patch must be non-empty when candidate_edit describes a code change`.
- Runtime commit `6cbc376 fix: harden invalid patch retry contract` strengthened the tool schema,
  prompt, retry hint, and validation error. Empty-patch decisions now get a clear two-mode contract:
  raw `diff --git` patch for code changes, or `No edit; ...` for bounded diagnostics only.
- The next loop produced a real `candidate_patch`, but it was based on hallucinated source context
  and `git apply --check` rejected it.
- Runtime commit `080c8dd feat: include candidate source context in planning` now adds bounded
  excerpts of current candidate source files to the planner prompt so raw diffs can use real local
  context.
- The next loop produced a more plausible patch, but `git apply --check` still rejected the diff
  as corrupt because the hunk line counts were wrong.
- Runtime commit `08a1a51 fix: tolerate recounted candidate patch hunks` keeps the existing path,
  symlink, binary, delete, rename, mode, and whitespace guards, but runs `git apply --recount` so
  generated diffs with wrong hunk counts can still be checked and applied when their content is
  otherwise valid.

Naive baseline score:

- After the recount fix, the loop made a valid no-edit diagnostic score:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_naive_attention_seed.py --seq-lens 128 --total-tokens 512 --num-heads 4 --head-dim 128 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`.
- The naive seed passed correctness in both causal modes.
- Noncausal case: max_abs_error `0.00390625`, `127.4438705444336` ms,
  `0.0010531516927932967` TFLOPS.
- Causal case: max_abs_error `0.015625`, `72.09878540039062` ms,
  `0.000930790492895549` TFLOPS.
- Naive geomean: `0.000990082614345315` TFLOPS.
- Gate result: rejected versus current best `0.10830947571120902` because this regressed geomean
  throughput.
- Runtime commit `5b267f5 docs: record naive baseline and patch recount` records the naive baseline
  and the updated patch substrate in `knowledge/ampere.md`.

Verification:

- For `6cbc376`, full lint passed, full unit suite passed with 131 tests, and `git diff --check`
  passed in `/home/ubuntu/avo-ampere`.
- For `080c8dd`, full lint passed, full unit suite passed with 131 tests, and `git diff --check`
  passed in `/home/ubuntu/avo-ampere`.
- For `08a1a51`, full lint passed, full unit suite passed with 132 tests, and `git diff --check`
  passed in `/home/ubuntu/avo-ampere`.
- The docs-only runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `5b267f525166f0733f2801f491106ceeebaa4cf4`.

Tradeoffs and decision:

- The source-excerpt prompt increases input size, but it directly addresses invalid patches caused
  by planning from filenames alone.
- `git apply --recount` does not bypass candidate path or patch-type restrictions; it only tolerates
  bad hunk lengths.
- The naive seed should remain a correctness reference, not a performance branch. The current
  lineage best remains `0.10830947571120902` geomean TFLOPS on nested lineage commit `07f1441`.

## 2026-05-08 - Checkpoint 3.35: Record invalid constant-only MMA head-dim extension

Success criteria for this checkpoint:

- Run one bounded loop after recording the naive baseline and updated patch substrate.
- Confirm that valid patches can pass application, run the bounded command, and clean up after gate
  rejection.
- Capture any concrete CUDA constraint discovered by the attempted patch.

Loop result:

- The agent produced a real `candidate_patch` against `candidates/cuda_mma_attention_seed.py` and
  `candidates/cuda_mma_attention/attention_kernel.cu`.
- Patch application succeeded through the guarded `git apply --recount` substrate.
- The patch only changed `SMOKE_HEAD_DIM` and `kHeadDim` from 16 to 32, then ran:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_mma_attention_seed.py --seq-lens 32 --total-tokens 32 --num-heads 1 --head-dim 32 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`.
- Scoring failed correctness because the extension failed to compile for the noncausal case and
  the causal case still hit the wrapper/runtime head-dim guard.
- Gate result: rejected with candidate geomean `0.0` versus current best
  `0.10830947571120902`.
- Cleanup result: checked reverse patch application succeeded, leaving the runtime worktree clean.

CUDA finding:

- WMMA does not support the attempted fragment shapes:
  `fragment<matrix_a, 16, 16, 32, ...>`,
  `fragment<matrix_b, 16, 16, 32, ...>`,
  `fragment<accumulator, 16, 16, 32, ...>`, and
  `fragment<accumulator, 16, 32, 16, ...>` were incomplete/unsupported.
- A head-dimension-32 MMA extension must keep the WMMA K dimension at 16 and explicitly process
  two 16-wide chunks for QK and PV. Do not repeat the constant-only `kHeadDim=32` patch.

Reliability note:

- This is the first post-hardening loop where a generated patch applied, the bounded score command
  executed, and rejected patch cleanup succeeded. The patch was wrong, but the orchestration path
  worked end to end.
- Runtime commit `3fb4724 docs: record invalid mma head-dim extension` records this in
  `knowledge/ampere.md`.

Verification:

- The runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `3fb472451aaf6cf027bca082087b9e52054e75cb`.

Tradeoffs and decision:

- The next MMA patch should be a real two-fragment implementation, not a constant change.
- The current lineage best remains `0.10830947571120902` geomean TFLOPS on nested lineage commit
  `07f1441`.

## 2026-05-08 - Checkpoint 3.36: Block recorded no-patch compile repeats

Success criteria for this checkpoint:

- Run one bounded loop after recording the invalid constant-only MMA head-dim patch.
- If the agent falls back to a known compile-only diagnostic, make that repeat invalid unless it is
  build-checking a real patch.

Loop result before the fix:

- The agent chose a no-edit compile diagnostic for
  `candidates/cuda_warp_rows_attention/attention_kernel.cu`.
- The compile succeeded and reproduced the already-recorded ptxas baseline:
  BF16/Half entry points at 48 registers, 1 barrier, 16896 bytes shared memory, no spills; FP32 at
  56 registers, 1 barrier, 33280 bytes shared memory, no spills.
- No patch was applied, no score was run, and no lineage commit was created.

Reliability fix:

- Runtime commit `05afd39 fix: reject recorded compile baselines` rejects no-patch compile
  diagnostics for sources whose ptxas baselines are already recorded:
  `candidates/cuda_warp_rows_attention/attention_kernel.cu` and
  `candidates/cuda_tiled_attention/attention_kernel.cu`.
- The same sources can still be compiled when a non-empty `candidate_patch` is present, so patch
  build-checks remain allowed.
- The repo context now tells the planner that these no-patch compile diagnostics are already
  recorded.

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/agent.py tests/test_agent.py` passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_agent.py` passed, 49 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 133 tests.
- Whitespace:
  `git diff --check` passed in `/home/ubuntu/avo-ampere`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `05afd390a358308e2601da06519e3eb2ff203b54`.

Tradeoffs and decision:

- This guard is intentionally narrow. It blocks repeated known no-patch compile baselines while
  preserving compile as a build-check for real candidate patches.
- The current lineage best remains `0.10830947571120902` geomean TFLOPS on nested lineage commit
  `07f1441`.

## 2026-05-08 - Checkpoint 3.37: Record unsafe cp.async patch shape

Success criteria for this checkpoint:

- Run one bounded loop after blocking recorded no-patch compile baselines.
- If the agent proposes a patch, determine whether it reaches compile/score or reveals another
  invalid patch pattern.

Loop result:

- The agent proposed a `candidate_patch` for
  `candidates/cuda_warp_rows_attention/attention_kernel.cu` and selected a patched compile check:
  `uv run --extra cuda python -m avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu --out-dir build/warp_cpasync_check`.
- Patch validation rejected the raw diff before compilation:
  `git apply --check failed` with trailing whitespace and corrupt hunk structure.
- No command ran, no score was produced, and no lineage commit was created.

CUDA finding:

- The proposed structure was not a viable cp.async direction even aside from patch formatting.
- It issued 16-byte `cp.async.cg.shared.global` copies at scalar element positions, which risks
  overlap and misalignment.
- It removed the previous zero-fill behavior for out-of-tile lanes.
- It waited immediately with `cp.async.wait_group 0` and `__syncthreads()`, so it did not introduce
  real double-buffered overlap.

Knowledge update:

- Runtime commit `769c5ca docs: record unsafe cpasync patch shape` records that future cp.async
  patches need vector-aligned 16-byte groups, preserved zero-fill or guarded shared-memory state
  for partial tiles, and a real overlapped pipeline rather than a single-stage copy/wait replacement.

Verification:

- The runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `769c5caa68e8cb068afcd98e0d8f2a8f55181767`.

Tradeoffs and decision:

- The compile-repeat guard successfully forced the planner away from the known no-patch ptxas
  baseline and toward a real patch attempt.
- The next cp.async attempt should be double-buffered and vector-aligned, or the planner should move
  to a different patch direction.
- The current lineage best remains `0.10830947571120902` geomean TFLOPS on nested lineage commit
  `07f1441`.

## 2026-05-08 - Checkpoint 3.38: Record partial two-chunk MMA patch failure

Success criteria for this checkpoint:

- Run one bounded loop after recording the unsafe cp.async patch shape.
- See whether the planner can produce the intended two-16-wide MMA head-dim-32 patch.

Loop result:

- The agent produced a two-chunk `candidate_patch` for the MMA seed and selected a head-dim-32
  score command.
- Patch application succeeded, the bounded score command ran, and rejected-patch cleanup succeeded.
- The noncausal case failed during CUDA compilation.
- The causal case also failed, reporting the old head-dim guard after the failed extension build.
- Gate result: rejected with candidate geomean `0.0` versus current best
  `0.10830947571120902`.

Compile failure:

- The patch correctly moved toward 16-wide WMMA chunks, but only partially converted the original
  `kHeadDim` constant.
- Stale `linear / kHeadDim` expressions remained after `kHeadDim` was removed.
- The final output loop redeclared `row` after adding a new `linear / head_dim` calculation.
- The patch also needed a clearer separation between score tile size (`16x16`) and output tile size
  (`16xhead_dim`) so `pv_tile` and `output_acc` are sized/indexed for head_dim 32.

Knowledge update:

- Runtime commit `0ccf909 docs: record partial mma two-chunk failure` records that a correct
  two-chunk patch must consistently use runtime `head_dim` for row/dim indexing and global strides,
  size output accumulators for the maximum supported head dimension, and avoid duplicate local
  declarations in the final store loop.

Verification:

- The runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `0ccf9090a40d9d656e1480709df2dc88e2e37adf`.

Tradeoffs and decision:

- This is still a useful failure: the planner is now targeting the right WMMA decomposition, and the
  remaining issue is ordinary C++/indexing completeness.
- The current lineage best remains `0.10830947571120902` geomean TFLOPS on nested lineage commit
  `07f1441`.

## 2026-05-08 - Checkpoint 3.39: Accept warp-row seq256 smoke

Success criteria for this checkpoint:

- Run one bounded loop after recording the partial two-chunk MMA failure.
- Accept a candidate only if correctness passes and geomean improves the lineage best.
- Align runtime wrapper caps and planner validation with any accepted larger smoke envelope.

Loop result:

- The agent patched `candidates/cuda_warp_rows_attention_seed.py`, changing
  `MAX_SMOKE_SEQUENCE` from 128 to 256.
- The CUDA kernel was unchanged; the patch validated that the existing warp-row global path handles
  a larger BF16 smoke shape at head_dim 128.
- Score command:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens 256 --total-tokens 1024 --num-heads 4 --head-dim 128 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`.
- Noncausal case: correct, max_abs_error `0.001953125`, `1.0882560014724731` ms,
  `0.4933314507556887` TFLOPS.
- Causal case: correct, max_abs_error `0.015625`, `0.8223999738693237` ms,
  `0.3264049909158355` TFLOPS.
- Candidate geomean: `0.4012802607933843` TFLOPS.
- Gate result: accepted versus prior best `0.10830947571120902`.
- Nested lineage accepted commit: `cfe5b45cdbe31e3794d5fea0eb55d7d4d1db1e24`.

Runtime updates:

- Runtime commit `f4da265 feat: accept warp row seq256 smoke` commits the accepted wrapper cap,
  updates the planner's no-patch warp-row score cap to `seq_lens<=256`,
  `head_dim<=128`, `total_tokens<=1024`, `num_heads<=4`, and records the new
  best in `knowledge/ampere.md`.
- Historical rejected baselines in knowledge now say `then-current` where they reference the old
  `0.10830947571120902` best.

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/agent.py tests/test_agent.py candidates/cuda_warp_rows_attention_seed.py` passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_agent.py` passed, 49 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 133 tests.
- Whitespace:
  `git diff --check` passed in `/home/ubuntu/avo-ampere`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `f4da265e6f4ed5e07d9ab2accb5a1bedce8d9bbd`.

Tradeoffs and decision:

- This is a larger validated smoke envelope, not a new CUDA optimization. It is still valuable
  because it gives the agent a higher-throughput accepted baseline and proves the warp-row global
  path scales to `seq_len=256`.
- The current lineage best is now `0.4012802607933843` geomean TFLOPS on nested lineage commit
  `cfe5b45`.

## 2026-05-08 - Checkpoint 3.40: Gate on benchmark case signature

Success criteria for this checkpoint:

- Audit the accepted seq256 result for gate quality.
- Prevent future candidates from winning purely by changing the scored workload.

Issue found:

- The seq256 warp-row result is valid and useful, but it also exposed a gate weakness: TFLOPS
  improved partly because the workload changed from seq128/512 tokens to seq256/1024 tokens.
- That kind of shape-only change should establish a new baseline, but future throughput comparisons
  must be against the same benchmark case signature.

Reliability fix:

- Runtime commit `622582e fix: gate on benchmark case signature` makes `commit_score` load the
  current `scores/latest.json` payload and compare sorted scored-case signatures before accepting
  a candidate.
- A candidate must now pass correctness, have finite positive geomean, and match the current best's
  benchmark cases before geomean comparison can accept it.
- Runtime commit `2f7fc68 docs: clarify fixed-case gate` records the rule in `knowledge/ampere.md`.

Verification:

- Focused lint:
  `uv run --extra dev ruff check avo/lineage.py tests/test_lineage.py` passed.
- Focused tests:
  `uv run --extra dev pytest tests/test_lineage.py` passed, 10 tests.
- Full lint:
  `uv run --extra dev ruff check .` passed.
- Full unit suite:
  `uv run --extra dev pytest` passed, 135 tests.
- Whitespace:
  `git diff --check` passed in `/home/ubuntu/avo-ampere`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `2f7fc68ff08aa1a44194bc25937cd8173befa7b3`.

Tradeoffs and decision:

- The current accepted seq256 lineage stays as the active comparison suite. Future optimization
  candidates need to score that same case set unless we deliberately reseed the benchmark suite.
- This makes the throughput gate stricter and prevents shape-only TFLOPS inflation.

## 2026-05-08 - Checkpoint 3.41: Verify fixed-case gate rejects seq512 shape-only score

Success criteria for this checkpoint:

- Run one bounded loop with the fixed-case gate active.
- Confirm shape-only workload growth no longer enters lineage even if TFLOPS increases.

Loop result:

- The agent patched `MAX_SMOKE_SEQUENCE` from 256 to 512 in
  `candidates/cuda_warp_rows_attention_seed.py` and scored:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_warp_rows_attention_seed.py --seq-lens 512 --total-tokens 2048 --num-heads 4 --head-dim 128 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`.
- The score passed correctness in both causal modes.
- Noncausal case: max_abs_error `0.001953125`, `2.167423963546753` ms,
  `0.9907999930414525` TFLOPS.
- Causal case: max_abs_error `0.015625`, `1.4129600524902344` ms,
  `0.7599236950171464` TFLOPS.
- Candidate geomean: `0.8677167693061046` TFLOPS.
- Gate result: rejected despite higher TFLOPS because benchmark cases differed from the current
  seq256 best.
- Cleanup result: checked reverse patch application succeeded and restored the runtime worktree.

Knowledge update:

- Runtime commit `88e61bd docs: record rejected shape-only seq512 score` records that seq512 passed
  correctness but must not be treated as a gate improvement unless the benchmark suite is
  deliberately reseeded.

Verification:

- The runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `88e61bd3ccb0513264c8033834fdddc8f3cb73c2`.

Tradeoffs and decision:

- The fixed-case gate did exactly what it should: kept lineage at `cfe5b45` / `0.4012802607933843`
  while still preserving useful correctness information about seq512.
- The current optimization target remains the accepted seq256 case signature.

## 2026-05-08 - Checkpoint 3.42: Record compiled but incorrect MMA head-dim32 attempt

Success criteria for this checkpoint:

- Run one bounded loop after the seq512 fixed-case rejection.
- Capture whether the next MMA two-chunk attempt compiles and whether it passes correctness.

Loop result:

- The agent produced another two-chunk MMA `candidate_patch` for
  `candidates/cuda_mma_attention_seed.py` and `candidates/cuda_mma_attention/attention_kernel.cu`.
- Patch application succeeded and the score command ran:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_mma_attention_seed.py --seq-lens 32 --total-tokens 32 --num-heads 1 --head-dim 32 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`.
- Unlike the previous two-chunk attempt, this one compiled and executed.
- It failed correctness in both causal modes.
- Noncausal case: max_abs_error `86.6435546875`, `0.6364799737930298` ms, TFLOPS `0.0`.
- Causal case: max_abs_error `218.828125`, `0.9138879776000977` ms, TFLOPS `0.0`.
- Gate result: rejected for failed correctness.
- Cleanup result: checked reverse patch application succeeded.

CUDA finding:

- The patch widened `pv_tile` to `kTile * kHeadDim`, but left `output_acc` at `kTileElements`
  while loops wrote `kTile * head_dim` elements.
- That likely corrupted shared memory and explains the large tolerance failures.
- A correct two-chunk MMA attempt must widen both `pv_tile` and `output_acc`, keep score and
  probability tiles at `16x16`, and verify the PV store offsets for both 16-wide output chunks.

Knowledge update:

- Runtime commit `f8a81b2 docs: record mma two-chunk correctness failure` records this failure mode
  in `knowledge/ampere.md`.

Verification:

- The runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `f8a81b2860fa4160b8676eef33af399750b83f5a`.

Tradeoffs and decision:

- The MMA path is making progress from compile failure to runtime correctness failure, but it is
  still not ready to challenge the seq256 warp-row baseline.
- The current lineage best remains `0.4012802607933843` geomean TFLOPS on nested lineage commit
  `cfe5b45`.

## 2026-05-08 - Checkpoint 3.43: Record malformed MMA widened-row patch

Success criteria for this checkpoint:

- Run one bounded loop after recording the compiled-but-incorrect MMA head-dim32 attempt.
- Capture whether the next patch fixes the widened output accumulator issue.

Loop result:

- The agent proposed another MMA head-dim32 patch and selected the same bounded score command.
- Patch validation rejected the raw diff before compile: trailing whitespace and corrupt hunk
  structure.
- No score was produced and no lineage commit was created.

CUDA finding:

- The proposed patch did widen both `pv_tile` and `output_acc`, but it also introduced the wrong
  row-indexing expression `linear / kTile` for widened output loops.
- For a row-major 16x32 output tile, row stride is the head dimension. Row indexing should use
  `linear / head_dim` or `linear / kHeadDim`, not `linear / kTile`.

Knowledge update:

- Runtime commit `8efc825 docs: record mma widened-row indexing guard` records this guardrail in
  `knowledge/ampere.md`.

Verification:

- The runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `8efc825e3fe0e6d36d6ae63804d7f881caf28c1f`.

Tradeoffs and decision:

- The failure was patch-formatting, but the proposed indexing would have been a correctness bug even
  if the diff applied.
- The current lineage best remains `0.4012802607933843` geomean TFLOPS on nested lineage commit
  `cfe5b45`.

## 2026-05-08 - Checkpoint 3.44: Record MMA PV chunk offset failure

Success criteria for this checkpoint:

- Run one bounded loop after recording the widened-row indexing guard.
- Capture whether the next two-chunk MMA head-dim32 attempt passes correctness.

Primer and research context:

- Re-read the active architecture notes and runtime knowledge base using the repo-primer workflow.
- Refreshed Exa search against primary Ampere FlashAttention sources.
- The same source direction remains valid: CUTLASS CuTeDSL Ampere FlashAttention v2 uses 128-bit
  `cp.async`, Ampere BF16/FP16 tensor-core MMA with `16x8x16` atoms, register pipelining, online
  softmax rescaling, and head-dim padding/alignment. Dao-AILab's SM80 forward path reinforces the
  need for explicit `cp.async` staging, bounded predicates, and separate QK/PV MMA handling.

Loop result:

- The agent proposed a two-chunk MMA head_dim32 patch and this time the patch applied cleanly.
- The selected score command was:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_mma_attention_seed.py --seq-lens 32 --total-tokens 32 --num-heads 1 --head-dim 32 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`.
- The score command completed without a compile failure, but both cases failed correctness.
- Noncausal case: max_abs_error `1.748046875`, `0.4551360011100769` ms, TFLOPS `0.0`.
- Causal case: max_abs_error `2.546875`, `0.5319679975509644` ms, TFLOPS `0.0`.
- Gate result: rejected for failed correctness.
- Cleanup result: checked reverse patch application succeeded.

CUDA finding:

- The patch widened both `pv_tile` and `output_acc`, which fixed the previous likely shared-memory
  overrun class.
- The remaining bug is likely the PV chunk store offset: it used
  `&pv_tile[chunk * kTile * 16]` while storing each 16-wide output fragment with leading dimension
  `kHeadDim == 32`.
- In a row-major 16x32 output tile, chunk 1 is a column offset, not a later row block. The chunk
  store should be shaped like `&pv_tile[chunk * 16]` with leading dimension `kHeadDim`.

Knowledge update:

- Runtime commit `a9274e6 docs: record mma pv chunk offset guard` records this guardrail in
  `knowledge/ampere.md`.

Verification:

- The runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `a9274e67d48455df50b673a28320b67a69a568ae`.

Tradeoffs and decision:

- This is progress from corrupt diff and large correctness failures to a compiling patch with smaller
  tolerance error, but it is still not lineage-eligible.
- The current lineage best remains `0.4012802607933843` geomean TFLOPS on nested lineage commit
  `cfe5b45`.

## 2026-05-08 - Checkpoint 3.45: Discourage repeated MMA baseline smoke

Success criteria for this checkpoint:

- Run one bounded loop after recording the PV chunk offset guard.
- Check whether the agent uses the new guardrail to attempt the corrected head_dim32 MMA patch.

Loop result:

- The agent made no code edit.
- It reran the current MMA seed at its already-supported maximum smoke shape:
  `seq_len=32`, `head_dim=16`, `total_tokens=32`, `num_heads=1`, BF16, both causal modes.
- The score passed correctness in both modes.
- Noncausal case: max_abs_error `0.001953125`, `0.5936319828033447` ms,
  `0.00011039836447240482` TFLOPS.
- Causal case: max_abs_error `0.00390625`, `0.5804799795150757` ms,
  `5.6449836611718976e-05` TFLOPS.
- Geomean TFLOPS: `7.894282511202798e-05`.
- Gate result: rejected because the candidate benchmark cases differ from the current seq256
  warp-row best.

Decision-contract finding:

- This was a valid diagnostic, but it spent a full loop on a known no-edit MMA baseline that was
  already recorded as too small and not lineage-eligible.
- Future no-edit MMA baseline scores should be reserved for environment/toolchain checks, not normal
  candidate-improving steps.

Knowledge update:

- Runtime commit `66474bb docs: discourage repeated mma baseline smoke` records this in
  `knowledge/ampere.md`.

Verification:

- The runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `66474bb3f2f7721619bef49b8d7347a387f809e4`.

Tradeoffs and decision:

- The MMA seed remains a useful correctness foothold, but the active improvement path should return
  to patching the head_dim32 two-chunk implementation or move to a fixed-case improvement on the
  seq256 warp-row suite.
- The current lineage best remains `0.4012802607933843` geomean TFLOPS on nested lineage commit
  `cfe5b45`.

## 2026-05-08 - Checkpoint 3.46: Record cp.async pipeline compile guard

Success criteria for this checkpoint:

- Run one bounded loop after discouraging the repeated MMA baseline smoke.
- Capture whether the agent returns to an edit that can improve the current fixed-case lineage.

Loop result:

- The agent switched back to the warp-row seed and proposed double-buffered `cp.async` K/V staging.
- The patch applied cleanly, then selected a compile-only check:
  `uv run --extra cuda python -m avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu --out-dir build/warp_async_check`.
- Compilation failed before scoring.
- The relevant compile errors were:
  - `identifier "__pipeline_commit" is undefined`
  - `identifier "__pipeline_wait_prior" is undefined`
- Gate result: no gate decision because no score was produced.
- Cleanup result: checked reverse patch application succeeded.

CUDA finding:

- The patch used CUDA pipeline intrinsics that are not available in the current candidate compile
  path without first proving the required include/API contract.
- Future cp.async attempts should either use a known-good inline PTX helper or the standard CUDA
  pipeline API with a tiny compile smoke proving the exact include and syntax first.
- The patch also treated a 16-byte BF16 async copy as if it covered 16 elements. For BF16, 16 bytes
  covers 8 elements, so vector groups should align on 8-element boundaries.
- The proposed copy loop mixed scalar fallback writes into the same shared-memory region that an
  async vector copy could still be writing. Full 16-byte vector groups and scalar tails must be
  disjoint until the wait/commit protocol is correct.

Knowledge update:

- Runtime commit `d541802 docs: record cpasync pipeline compile guard` records these constraints in
  `knowledge/ampere.md`.

Verification:

- The runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `d54180241aabb06f3a695dddd519eeae4b0639d7`.

Tradeoffs and decision:

- This was a useful strategic pivot away from repeated MMA baseline diagnostics, but the cp.async
  implementation still needs a smaller compile-proven primitive before touching the warp-row hot
  path again.
- The current lineage best remains `0.4012802607933843` geomean TFLOPS on nested lineage commit
  `cfe5b45`.

## 2026-05-08 - Checkpoint 3.47: Record CUDA pipeline primitive include

Success criteria for this checkpoint:

- Refresh Ampere `cp.async` source context after the failed `__pipeline_*` compile.
- Record the concrete include/API path that can make the next cp.async attempt compile.

Research result:

- Exa search found NVIDIA's CUDA Programming Guide asynchronous-copy section and PTX ISA docs as
  the relevant primary references for global-to-shared async copies.
- Local CUDA 13 headers under the Python-provided CUDA root contain the immediately actionable
  candidate compile contract.
- `cuda_pipeline_primitives.h` declares `__pipeline_memcpy_async`, `__pipeline_commit`, and
  `__pipeline_wait_prior`.
- The helper routes 16-byte async copies to `cp.async.cg.shared.global`, supports source-size /
  zero-fill handling for partial copies, and checks shared/global address spaces plus 4/8/16-byte
  alignment.

Knowledge update:

- Runtime commit `dd50425 docs: record cuda pipeline primitive include` records that future cp.async
  attempts can first add `#include <cuda_pipeline_primitives.h>` and compile a tiny candidate-local
  smoke before restructuring the warp-row loop.

Verification:

- The runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `dd50425b1745ae8a06f732774cc15ad24899448d`.

Tradeoffs and decision:

- This does not make the failed warp-row patch correct; it only removes the immediate undefined
  intrinsic blocker for a future, smaller cp.async smoke.
- The 8-BF16-elements-per-16-byte-copy and disjoint scalar-tail constraints from Checkpoint 3.46
  still apply.

## 2026-05-08 - Checkpoint 3.48: Reject unchanged-source timing wins

Success criteria for this checkpoint:

- Run one bounded loop after recording the CUDA pipeline primitive include.
- Detect and fix any gate issue exposed by the next accepted or rejected attempt.

Loop result:

- The agent made no code edit.
- It reran the existing warp-row seed on the fixed seq256/head_dim128 BF16 case suite.
- The rerun passed correctness and measured faster than the prior best:
  - Noncausal: max_abs_error `0.001953125`, `0.9894400238990784` ms,
    `0.5426007630905784` TFLOPS.
  - Causal: max_abs_error `0.015625`, `0.6876479983329773` ms,
    `0.3903675378256776` TFLOPS.
  - Geomean: `0.460232249967343`.
- The gate accepted this even though the source snapshot was unchanged. This was a timing-noise
  acceptance, not a real kernel variation.

Runtime fix:

- Added a source-snapshot equality gate in `commit_score`.
- If `candidate_patch` is empty and the current candidate source snapshot matches
  `sources/latest` from the lineage head, the gate now rejects with
  `candidate source is unchanged from current best`.
- Added regression coverage for the unchanged-source rerun case.
- Updated the runtime knowledge base to record that identical-source reruns must not advance lineage.

Verification:

- `uv run --extra dev pytest tests/test_lineage.py`: `11 passed`.
- `uv run --extra dev pytest`: `136 passed`.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `b101ad682e141750d956a831b0f5e3ab41dc0210`.

Lineage correction:

- The no-edit accepted lineage commit was `adccd99 evolve: accept candidate`.
- Reverted it with a normal lineage commit, `e1ca520 Revert "evolve: accept candidate"`, so the
  latest lineage score again reflects the last source-changing accepted candidate.
- Current nested lineage score is restored to `0.4012802607933843` geomean TFLOPS:
  noncausal `0.4933314507556887` TFLOPS and causal `0.3264049909158355` TFLOPS.

Tradeoffs and decision:

- The faster no-edit sample is useful as noise evidence, but not as evolutionary progress.
- Future accepted candidates must now both match the benchmark case signature and change the source
  snapshot unless they carry a non-empty accepted patch.

## 2026-05-08 - Checkpoint 3.49: Block repeated MMA compile diagnostics

Success criteria for this checkpoint:

- Run one bounded loop after the unchanged-source scoring gate fix.
- Prevent repeated no-edit diagnostics from consuming additional evolution steps when they only
  restate an already-known baseline.

Loop result:

- The agent made no code edit and ran a compile-only diagnostic:
  `avo compile --source candidates/cuda_mma_attention/attention_kernel.cu --out-dir
  build/mma_baseline_check`.
- The current MMA seed compiled successfully for `sm_86`.
- ptxas reported 40 registers, 1 barrier, and 3776 bytes of shared memory.
- No candidate was scored and no lineage gate decision was made. The nested lineage head remained
  `e1ca520057c7172e15ac8d58a9a4e8cb1924e57e`.

Runtime fix:

- Added `candidates/cuda_mma_attention/attention_kernel.cu` to the recorded no-patch compile
  diagnostic blocklist.
- Updated the repo-context prompt to tell the agent that MMA, warp-row, and tiled no-patch compile
  diagnostics are already recorded and should only be compiled when build-checking a non-empty
  `candidate_patch`.
- Added a regression test that rejects repeated no-patch MMA compile diagnostics.
- Updated the runtime knowledge base with the successful MMA compile details and the new policy.

Research refresh:

- Exa found NVIDIA's current CUDA Programming Guide async-copy and pipeline sections plus the
  NVIDIA Ampere tuning guide as the primary references for this area.
- Relevant constraints remain: LDGSTS / `cp.async` is for global-to-shared copies on CC 8.0+,
  supports 4/8/16-byte copies, gets L1-bypass behavior with 16-byte copies, requires matching
  alignment, benefits from 128-byte alignment, and needs a completion wait plus `__syncthreads()`
  before other threads consume prefetched shared data.
- CUDA pipeline commits should be invoked by converged warps, or waits can over-wait due warp
  entanglement.

Verification:

- `uv run --extra dev pytest tests/test_agent.py`: `50 passed`.
- `uv run --extra dev pytest`: `137 passed`.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `1fe4bc6baeb8aa4db95ec7f94bba8cff18c7f2a4`.

Tradeoffs and decision:

- The successful compile confirms the MMA seed still builds, but it is not evolutionary progress.
- Future MMA work should patch the wrapper/kernel first, then compile or score that changed source.

## 2026-05-08 - Checkpoint 3.50: Tighten candidate diff hygiene

Success criteria for this checkpoint:

- Run another bounded loop after blocking repeated MMA compile diagnostics.
- If the next attempt produces a source-changing patch, ensure malformed diffs are rejected cleanly
  and record the reliability lesson.

Loop result:

- The agent attempted a source-changing head_dim32 MMA patch.
- The intended direction matched the current knowledge: keep WMMA fragments at 16x16x16, process
  two 16-wide QK chunks, widen `pv_tile` and `output_acc`, and use PV column offsets like
  `&pv_tile[chunk * 16]`.
- The patch was rejected before execution by `git apply --check`.
- Rejection details: trailing whitespace on two added blank lines and `error: corrupt patch at line
  160`.
- No source was modified, no score ran, and the nested lineage head remained
  `e1ca520057c7172e15ac8d58a9a4e8cb1924e57e`.

Runtime fix:

- Tightened the Anthropic variation prompt and tool schema to say candidate patches must use exact
  current file context, valid hunk structure, and whitespace-clean added lines.
- Added prompt/context regression assertions so the guidance stays present.
- Updated runtime knowledge to record the malformed head_dim32 MMA attempt and recommend smaller,
  compile-checkable structural slices.

Research refresh:

- Exa found NVIDIA's BF16 Tensor Core GEMM CUDA sample. The relevant takeaways for future MMA work
  are that Ampere BF16 WMMA uses 16x16x16 fragments, shared-memory skew/padding is used to reduce
  WMMA load bank conflicts, and CUDA pipeline async copies can reduce register pressure for
  global-to-shared staging.

Verification:

- `uv run --extra dev pytest tests/test_agent.py`: `50 passed`.
- `uv run --extra dev pytest`: `137 passed`.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `f87bd9c195a5e5f15e746c29eb71e7988b631cd0`.

Tradeoffs and decision:

- The orchestrator behaved correctly by rejecting the malformed patch without touching the tree.
- The next MMA attempt should be smaller than the rejected patch, preferably compile-checking only
  the first structural slice before changing the score shape.

## 2026-05-08 - Checkpoint 3.51: Require compile-first MMA shape extensions

Success criteria for this checkpoint:

- Run another bounded loop after the diff-hygiene prompt update.
- Stop repeated malformed MMA shape-extension attempts from jumping straight to score.

Loop result:

- The agent attempted another source-changing head_dim32 MMA patch.
- The patch again bundled wrapper changes, QK changes, PV changes, and a score command in one step.
- `git apply --check` rejected it before execution with `error: corrupt patch at line 120`.
- No source was modified, no score ran, and the nested lineage head remained
  `e1ca520057c7172e15ac8d58a9a4e8cb1924e57e`.

Additional finding:

- Beyond the malformed hunk, the proposed patch introduced undefined `head_dim` identifiers inside
  the CUDA kernel and still used unsupported WMMA fragment template shapes for the intended
  head_dim32 structure.
- This confirms the next MMA step should not jump straight to score. It needs a compile-only
  build-check of a smaller structural slice first.

Runtime fix:

- Added validator logic that rejects patched MMA seed score commands beyond head_dim16 unless the
  next command is an `avo compile` build-check first.
- Added regression coverage for patched MMA shape-extension score commands.
- Updated repo-context prompt text and runtime knowledge with the compile-first policy.

Verification:

- `uv run --extra dev pytest tests/test_agent.py`: `51 passed`.
- `uv run --extra dev pytest`: `138 passed`.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `e73681a8d209b8d8094c45cf12f2825f29e0d5b2`.

Tradeoffs and decision:

- This slows down risky MMA shape experiments by one loop, but prevents expensive score attempts
  from being selected before the candidate even compiles.
- Future head_dim32 MMA attempts should use `avo compile --source
  candidates/cuda_mma_attention/attention_kernel.cu --out-dir build/<name>` as the first command.

## 2026-05-08 - Checkpoint 3.52: Verify compile-first guard path

Success criteria for this checkpoint:

- Run one bounded loop after adding the compile-first guard for patched MMA shape extensions.
- Confirm the next agent decision uses a compile-checkable patch rather than another immediate
  patched head_dim32 score.

Loop result:

- The agent selected a compile-first patched step instead of a patched MMA score.
- The patch added a small WMMA foothold to the warp-row kernel: include `<mma.h>`, add shared
  16x16 Q/K staging, and compile a BF16 `head_dim == 16` score-tile path.
- The patch applied cleanly and `avo compile --source
  candidates/cuda_warp_rows_attention/attention_kernel.cu --out-dir build/warp_mma_check`
  succeeded.
- Cleanup reverted the patch afterward because compile-only attempts do not enter lineage.
- The nested lineage head remained `e1ca520057c7172e15ac8d58a9a4e8cb1924e57e`.

Compile diagnostics:

- BF16 entry point: 48 registers, 1 barrier, 17984 bytes shared memory, no spills.
- FP16 entry point: 48 registers, 1 barrier, 16896 bytes shared memory, no spills.
- FP32 entry point: 56 registers, 1 barrier, 33280 bytes shared memory, no spills.

Decision:

- The compile-first guard worked, but the generated patch is not scoreable as-is.
- It only handles a `head_dim == 16` BF16 score path for `warp_id == 0`, does not integrate WMMA
  scores into the existing online softmax/output accumulation for all rows, and would leave that
  branch without a final output update.
- Runtime knowledge now records that future warp-row WMMA work should either keep the normal path
  intact while compiling an isolated helper, or fully route all rows through a correct online-softmax
  path before scoring.

Verification:

- Runtime knowledge update passed `git diff --check`.

- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `55499823850b22dda2061672d14d5238b0e21a7f`.

## 2026-05-08 - Checkpoint 3.53: Record WMMA fragment layout guard

Success criteria for this checkpoint:

- Research the WMMA fragment-indexing risk exposed by the compile-first warp-row MMA patch.
- Add the resulting guardrail to the runtime knowledge base.

Research result:

- Exa and the official CUDA C++ Programming Guide both point to the same rule: WMMA fragment
  internal storage is not a stable row/column layout.
- The guide says matrix elements are distributed across warp lanes and the mapping into fragment
  storage is unspecified and can change across architectures.
- `load_matrix_sync`, `mma_sync`, and `store_matrix_sync` are warp-wide operations; pointer,
  leading dimension, layout, and template parameters must be the same across warp lanes.

Runtime knowledge update:

- Do not infer row or column positions from `fragment.x[]`.
- Uniform per-element transforms on a fragment are acceptable, but row/column selection should use
  `wmma::store_matrix_sync` into memory with an explicit layout before indexing.
- This directly applies to the previous warp-row WMMA compile patch, which tried to pick row 0 from
  `score_frag.x[]`.

Verification:

- Runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `f8f41f9c939c406f87b57a4aca60d0aa1e1b49a3`.

## 2026-05-08 - Checkpoint 3.54: Verify CUDA pipeline primitive header smoke

Success criteria for this checkpoint:

- Run one bounded loop after adding the WMMA fragment layout guard.
- Prefer a compile-checkable, source-changing diagnostic over another no-edit baseline.

Loop result:

- The agent selected a compile-only cp.async header smoke for the warp-row kernel.
- The patch added `#include <cuda_pipeline_primitives.h>` and unused wrappers around
  `__pipeline_memcpy_async`, `__pipeline_commit`, and `__pipeline_wait_prior`.
- The patch applied cleanly and `avo compile --source
  candidates/cuda_warp_rows_attention/attention_kernel.cu --out-dir build/warp_cpasync_header_check`
  succeeded.
- Cleanup reverted the patch afterward because compile-only attempts do not enter lineage.
- The nested lineage head remained `e1ca520057c7172e15ac8d58a9a4e8cb1924e57e`.

Compile diagnostics:

- NVCC warned only that the commit/wait helper functions were unused.
- BF16 entry point: 48 registers, 1 barrier, 16896 bytes shared memory, no spills.
- FP16 entry point: 48 registers, 1 barrier, 16896 bytes shared memory, no spills.
- FP32 entry point: 56 registers, 1 barrier, 33280 bytes shared memory, no spills.

Decision:

- This proves the CUDA 13 header/API availability for local `__pipeline_*` primitives on sm86.
- It does not prove any performance improvement.
- The next cp.async attempt must still add real double-buffered overlap and keep 16-byte BF16
  groups aligned and disjoint from scalar tail writes.

Verification:

- Runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `4e683c0e4616c30caca3c4091ef364fe055231a7`.

## 2026-05-08 - Checkpoint 3.55: Double-buffer cp.async static shared-memory limit

Success criteria for this checkpoint:

- Run one bounded loop after the cp.async header smoke.
- Let the agent attempt a more concrete cp.async structural patch, then capture compile feedback.

Loop result:

- The agent attempted a double-buffered cp.async structural patch in the warp-row shared K/V staging
  path.
- The patch added a second K/V shared tile buffer and used `__pipeline_memcpy_async`,
  `__pipeline_commit`, and `__pipeline_wait_prior`.
- The patch applied cleanly, but compile failed.
- Cleanup reverted the patch afterward; no source remained modified and no lineage gate decision was
  made.
- The nested lineage head remained `e1ca520057c7172e15ac8d58a9a4e8cb1924e57e`.

Compile finding:

- BF16 and FP16 entry points reached ptxas with 56 registers, 1 barrier, 33280 bytes shared memory,
  and no spills.
- The FP32 entry point used 66048 bytes of static shared memory and failed ptxas:
  `uses too much shared data (0x10200 bytes, 0xc000 max)`.
- The full translation unit fails while the FP32 template instantiation is emitted, even if the
  target scoring path would be BF16.

Decision:

- Double-buffering the current static K/V tiles naively is not viable while the templated source
  emits FP32 with doubled buffers.
- Future double-buffering must avoid doubling FP32 static buffers, reduce the staged footprint, split
  dtype-specific kernels, or use dynamic shared memory plus the required launch attribute for
  above-48KB per-block allocation.

Verification:

- Runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `a75b2425de54c8ecbe7009019e42f0f69f361474`.

## 2026-05-08 - Checkpoint 3.56: Shared-memory opt-in and tile-width guardrails

Success criteria for this checkpoint:

- Research the static shared-memory limit exposed by the failed double-buffered cp.async compile.
- Record the exact sm86 rule and any cleanup needed after the rejected attempt.

Research result:

- Exa found NVIDIA's Ampere tuning guide and CUDA function-attribute documentation as the relevant
  primary references.
- On compute capability 8.6, a block can address up to 99 KB shared memory, but static shared memory
  remains limited to 48 KB for architectural compatibility.
- Above-48KB use requires dynamic shared memory plus explicit opt-in, and CUDA function attributes
  constrain requested dynamic shared memory plus static shared memory to the device opt-in limit.

Cleanup:

- After the rejected cp.async attempt, the runtime worktree showed an unaccepted residue:
  `kTileKeys = 64` in the warp-row kernel.
- That is unsafe with the current one-key-per-lane mapping: 32 lanes would process only keys 0..31
  while the tile loop advances by 64, skipping half the keys.
- Restored `kTileKeys = kWarpSize` and verified the source diff was clean before committing the
  knowledge update.

Runtime knowledge update:

- Do not propose static shared-memory allocations above 48 KB on sm86.
- Do not change `kTileKeys` above `kWarpSize` unless the score and V accumulation loops are changed
  to map multiple key columns per lane.

Verification:

- Runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `72facba19d863b51ed6a78e5dc8fe5cbc8743abd`.

## 2026-05-08 - Checkpoint 3.57: Dynamic shared-memory launch wiring gap

Success criteria for this checkpoint:

- Run one bounded loop after recording the sm86 shared-memory opt-in guardrails.
- Capture whether the agent can turn the static-buffer failure into a compile-checkable dynamic
  shared-memory structural patch.

Loop result:

- The agent proposed moving warp-row K/V staging from static shared arrays to `extern __shared__`.
- The patch applied cleanly and compiled successfully with
  `avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu --out-dir
  build/warp_dynamic_smem_check`.
- Cleanup reverted the patch afterward because compile-only attempts do not enter lineage.
- The nested lineage head remained `e1ca520057c7172e15ac8d58a9a4e8cb1924e57e`.

Compile diagnostics:

- Static shared memory dropped to 512 bytes for Half, BF16, and FP32 entry points.
- Half/BF16 used 48 registers, FP32 used 56 registers, with no spills.

Decision:

- The compile result is useful, but the generated patch is not scoreable as-is.
- It did not pass the required dynamic shared-memory byte count in the kernel launch configuration.
- It set `cudaFuncSetAttribute` only for the BF16 specialization.
- It computed dynamic shared-memory bytes using BF16 size unconditionally, which is wrong for FP32.
- Future dynamic-shared patches must wire both the launch third argument and dtype-specific
  `cudaFuncSetAttribute` values before scoring.

Verification:

- Runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `3316106482b916818a6dc8b42eb564c1625f1afe`.

## 2026-05-08 - Checkpoint 3.58: Rejected patch cleanup verification

Success criteria for this checkpoint:

- Close the reliability gap observed after a rejected compile attempt left a stray candidate source
  edit despite cleanup reporting success.
- Preserve the existing non-destructive cleanup model: reverse the candidate patch, verify, but do
  not force-reset user or agent changes.

Runtime change:

- Added a post-cleanup verification step for rejected candidate patches.
- After `git apply --reverse` succeeds, the runtime now checks the candidate patch paths with
  `git status --porcelain --untracked-files=all`.
- If any affected patch path remains dirty, cleanup is recorded as failed with
  `candidate patch cleanup left paths dirty`.
- Non-Git temporary test directories keep the old behavior because there is no `HEAD` to compare
  against.

Regression test:

- Added a Git-backed test where reverse-apply succeeds but command-side residue remains in the
  patched candidate file.
- The cleanup result is now marked failed, and the loop control path can stop on
  `cleanup_failed` instead of silently continuing from a contaminated source tree.

Verification:

- `uv run --extra dev pytest tests/test_evolve.py -q`: 35 passed.
- `uv run --extra dev pytest tests/test_cli.py -q`: 21 passed.
- `uv run --extra dev pytest`: 139 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `6e160162f6a27c9d57092b4c679686ce4a8e9826`.

## 2026-05-08 - Checkpoint 3.59: Scalar fallback unroll rejection

Success criteria for this checkpoint:

- Run one bounded loop after adding rejected-patch cleanup verification.
- Capture the result without allowing another malformed patch direction to repeat.

Loop result:

- The agent proposed adding `#pragma unroll` to the scalar fallback in
  `dot_product`.
- The candidate patch was rejected by `git apply --check`:
  `error: patch failed: candidates/cuda_warp_rows_attention/attention_kernel.cu:56`.
- No patch was applied, no cleanup was required, and no lineage gate decision was made.
- Runtime and lineage worktrees remained clean.

Decision:

- The generated patch referenced `qv`, `kv`, and `inner` in the scalar fallback, but those names
  only exist in the packed divisible-by-4 branch.
- The fixed benchmark head dimension is 128, so the current path uses the packed branch rather than
  the scalar fallback.
- Exa research found NVIDIA CUTLASS loop-unrolling guidance: unrolling is most relevant when trip
  counts are compile-time constants. The scalar fallback loop uses runtime `head_dim`.
- Do not spend another candidate on scalar fallback unrolling unless the benchmark suite includes an
  odd/non-packed head dimension.

Verification:

- Runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `bad2f88a18373adb240148f025b5f7e7544c021d`.

## 2026-05-08 - Checkpoint 3.60: Two-chunk MMA compile check

Success criteria for this checkpoint:

- Run one bounded loop after steering away from scalar fallback unrolling.
- Capture whether the agent can produce a smaller compile-checkable MMA head-dimension-32 slice.

Loop result:

- The agent generated a two-chunk MMA structural patch for
  `candidates/cuda_mma_attention/attention_kernel.cu`.
- The patch applied cleanly and compiled successfully with
  `avo compile --source candidates/cuda_mma_attention/attention_kernel.cu --out-dir
  build/mma_two_chunk_check`.
- Cleanup reverted the patch afterward because compile-only attempts do not enter lineage.
- The nested lineage head remained `e1ca520057c7172e15ac8d58a9a4e8cb1924e57e`.

Compile diagnostics:

- ptxas reported 40 registers, 1 barrier, 3776 bytes shared memory, and no spills for the BF16 MMA
  kernel on `sm_86`.

Decision:

- This is a useful compile-shape proof, not a correctness proof.
- The generated patch kept `kHeadDim == 16` and the 16x16 `pv_tile`/`output_acc` buffers.
- Its second Q/K chunk used `kHeadDim` as the global row stride, and its second PV chunk used
  leading dimension `kHeadDim * 2` into buffers sized only for 16x16.
- A real scoreable head_dim32 patch must widen `pv_tile` and `output_acc` to 16x32, use a 32-wide
  row stride, and store each 16-column PV chunk inside the widened row-major tile.

Verification:

- Runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `57dbb4bafe90c60c8d261a49649a684c777ca601`.

## 2026-05-08 - Checkpoint 3.61: Internal-32 MMA compile failure

Success criteria for this checkpoint:

- Run one bounded loop after the two-chunk MMA compile-shape proof.
- Capture whether a widened internal 16x32 MMA buffer patch compiles.

Loop result:

- The agent proposed an internal-32 MMA patch that widened `pv_tile` and `output_acc`, while keeping
  the wrapper at `kHeadDim == 16`.
- The patch applied and cleanup reverted it cleanly afterward.
- Compile failed with return code 2.

Compile failure:

- NVCC reported undefined identifiers in the QK block:
  `score_frag`, `q_frag`, and `k_frag`.
- The patch introduced `score_frag_0` and `score_frag_1` plus chunk-local Q/K fragments, but left the
  original single-fragment fill/load/mma lines in place after the new chunk loop.

Decision:

- This confirms the correct next repair is surgical: remove the stale single-fragment QK block after
  introducing the two-chunk loop, then store the accumulated two-chunk score fragment.
- Continue compile-first for this MMA direction; do not score until the widened buffers, row stride,
  QK accumulation, PV chunk stores, and wrapper shape are all consistent.

Verification:

- Runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `2217621afd7ff27e54037532c4545852239f7b70`.

## 2026-05-08 - Checkpoint 3.62: MMA cp.async malformed patch

Success criteria for this checkpoint:

- Run one bounded loop after recording the internal-32 MMA compile failure.
- Capture whether the agent stays on the surgical MMA compile repair or switches direction.

Loop result:

- The agent switched to an MMA single-stage cp.async patch for the 16x16 K tile.
- The candidate patch was rejected by `git apply --check` with
  `error: corrupt patch at line 53`.
- No patch was applied, no cleanup was required, and no lineage gate decision was made.

Decision:

- The proposed structure copied scalar BF16 elements with `__pipeline_memcpy_async` and immediately
  committed/waited before `wmma::load_matrix_sync`.
- That would not create a useful Ampere pipeline even if the diff had applied.
- Future MMA cp.async attempts must use aligned 16-byte groups, meaning 8 BF16 elements per group,
  keep scalar tails disjoint, and compile-check a clean diff before any score.

Verification:

- Runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `7d86b8491b3c3f3c1d0a97d790e910a40294838f`.

## 2026-05-08 - Checkpoint 3.63: q-offset hoist rejection

Success criteria for this checkpoint:

- Run one final bounded loop after recording the malformed MMA cp.async patch.
- Capture the result and leave both repos clean and pushed.

Loop result:

- The agent proposed hoisting a warp-row `q_row = q + q_offset` pointer for the global K/V path.
- The candidate patch was rejected by `git apply --check` because the context did not match the
  current source.
- No patch was applied, no cleanup was required, and no lineage gate decision was made.

Decision:

- The current warp-row kernel already computes `q_offset` before both tile loops.
- Introducing a `q_row` pointer is likely too small to beat timing noise, and the rejected patch also
  tried to branch on `can_stage_shared` before the source location where that value is declared.
- Do not repeat this as a candidate-improving direction without a larger, measurable address or load
  scheduling change.

Verification:

- Runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `c1589585a8fa9c7eaa4c1b721c79c50c5205909c`.

## 2026-05-08 - Checkpoint 3.64: Patch failure detail in attempt history

Success criteria for this checkpoint:

- Improve recovery from malformed candidate patches without changing the patch application trust
  boundary.
- Preserve enough detail in cross-step memory for the next agent decision to distinguish corrupt
  diffs, stale context, and other apply failures.

Research note:

- Exa search surfaced current agent patch-tool guidance emphasizing that failed patch application
  should return a clear, human-readable error string so the model can recover in the next step.
- This matches the observed local failure mode: attempt history only said `git apply --check failed`
  while the actionable details, such as `error: corrupt patch at line 53`, were buried in the raw
  JSON record.

Runtime change:

- Attempt-history summaries now include the patch result `stderr_tail` or `stdout_tail` for rejected
  candidate patches.
- Failed patch cleanup summaries also include the corresponding detail tail.
- Added a regression test that records `error: corrupt patch at line 53` and verifies the summary
  surfaces it to the next planning prompt.

Verification:

- `uv run --extra dev pytest tests/test_evolve.py -q`: 36 passed.
- `uv run --extra dev pytest`: 140 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `240935a9d1bac2e9f27946f029578be926ecaf75`.

## 2026-05-08 - Checkpoint 3.65: MMA unroll compile check

Success criteria for this checkpoint:

- Run one bounded loop after adding patch failure details to attempt history.
- Verify whether the richer failure detail helps the agent avoid another malformed patch.

Loop result:

- The agent produced a clean MMA patch adding `#pragma unroll` before several helper loops in
  `candidates/cuda_mma_attention/attention_kernel.cu`.
- The patch applied cleanly and compiled successfully with
  `avo compile --source candidates/cuda_mma_attention/attention_kernel.cu --out-dir
  build/mma_unroll_pragmas`.
- Cleanup reverted the patch afterward because compile-only attempts do not enter lineage.
- The nested lineage head remained `e1ca520057c7172e15ac8d58a9a4e8cb1924e57e`.

Compile diagnostics:

- ptxas reported 40 registers, 1 barrier, 3776 bytes shared memory, and no spills for the BF16 MMA
  kernel on `sm_86`, unchanged from the baseline MMA compile diagnostics.

Decision:

- The richer attempt-history detail appears to have helped steer away from another corrupt/stale
  diff, but the result is still compile-only.
- This patch is not a useful lineage candidate by itself because the MMA wrapper remains limited to
  the tiny seq32/head_dim16 case signature, which differs from the seq256/head_dim128 warp-row best.
- Do not repeat the same compile-only unroll step; any future MMA unroll score needs a deliberately
  reseeded MMA benchmark or a wrapper/kernel extension that can compete on the target suite.

Verification:

- Runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `1512a872dc62a6a9adca70bba64ee421d954901f`.

## 2026-05-08 - Checkpoint 3.66: Self-invalid patch guard and tiled rescale rejection

Success criteria for this checkpoint:

- Run one bounded loop after recording the MMA unroll compile-only result.
- Prevent decisions from executing when their own risk text says the patch is known invalid.

Loop result:

- The agent proposed a tiled online-softmax rescale patch.
- The patch was rejected by `git apply --check` because the context did not match the current tiled
  source.
- The decision's own risk text also said the patch left a stale `tile_scale` reference and would
  cause a compile error.
- No patch was applied, no cleanup was required, and no lineage gate decision was made.

Runtime change:

- Planner validation now rejects non-empty candidate patches whose decision text describes the patch
  as known invalid, currently including phrases such as `not ready to apply` or
  `will cause a compile error`.
- Added a regression test for a self-rejected tiled patch decision.

Tiled-kernel note:

- The correct online-softmax accumulation invariant remains:
  `output_acc = output_acc * old_scale + tile_acc * tile_scale`.
- `row_sum` must update as `row_sum = row_sum * old_scale + tile_sum * tile_scale`.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 52 passed.
- `uv run --extra dev pytest`: 141 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.
- Runtime push/fetch verification after code guard: local `main` and `origin/main` both resolved to
  `e78006143ec9388195b66d1db4511021b4770492`.
- Runtime push/fetch verification after knowledge update: local `main` and `origin/main` both
  resolved to `76e3aa7d75cef6086a3f70cdf2460b706cdeb8d7`.

## 2026-05-08 - Checkpoint 3.67: Warp-row packed dot-product unroll compile check

Success criteria for this checkpoint:

- Run one bounded loop after adding the self-invalid patch guard.
- Capture whether the next agent can produce a clean, compile-checkable patch.

Loop result:

- The agent produced a clean warp-row patch adding `#pragma unroll` before the packed 4-wide
  dot-product outer loop in `candidates/cuda_warp_rows_attention/attention_kernel.cu`.
- The patch applied cleanly and compiled successfully with
  `avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu --out-dir
  build/warp_unroll_dot_product`.
- Cleanup reverted the patch afterward because compile-only attempts do not enter lineage.

Compile diagnostics:

- BF16/Half stayed at 48 registers, 1 barrier, 16896 bytes shared memory, and no spills.
- FP32 rose to 64 registers, 1 barrier, 33280 bytes shared memory, and no spills.

Decision:

- This is a plausible low-risk score candidate on the fixed seq256/head_dim128 BF16 warp-row suite.
- The next loop should score the same patch rather than repeat compile-only.

Verification:

- Runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `50c085ea059a7336394a28c4ff2f696ffbb8cbf1`.

## 2026-05-08 - Checkpoint 3.68: Warp-row packed dot-product unroll regression

Success criteria for this checkpoint:

- Score the same warp-row packed dot-product unroll patch that compiled cleanly.
- Compare against the fixed seq256/head_dim128 BF16 warp-row lineage best.

Loop result:

- The agent re-applied the same `#pragma unroll` patch to the packed 4-wide dot-product outer loop.
- The patched candidate scored correctly on the fixed BF16 warp-row suite with three timing trials.
- The gate rejected the candidate because throughput regressed.
- Cleanup reverted the patch afterward; no lineage commit was made.

Score result:

- Noncausal: `0.4773985262900999` TFLOPS.
- Causal: `0.2881296921121001` TFLOPS.
- Geomean: `0.3708809652634344` TFLOPS.
- Current best remains `0.4012802607933843` geomean TFLOPS.

Decision:

- Do not repeat the packed dot-product outer-loop unroll as a candidate-improving step.
- The compile result was clean, but the BF16 score showed the unroll hint hurt aggregate throughput,
  mainly through the causal case.

Verification:

- Runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `68b742870989771f674b6e07c37e6037b85fe316`.

## 2026-05-08 - Checkpoint 3.69: Stale MMA patch rejection guard

Success criteria for this checkpoint:

- Run one bounded loop after closing the warp-row unroll regression.
- Prevent another self-described stale-code MMA patch from becoming an executed attempt.

Loop result:

- The agent proposed another head_dim32 two-chunk MMA patch.
- The patch was rejected by `git apply --check` because its context did not match the current MMA
  source.
- The patch also left stale single-chunk QK load/mma lines after the new two-chunk loop.
- The decision risk text identified the issue: stale QK load lines might reference undeclared
  fragments and must be removed.

Runtime change:

- Planner validation now also rejects non-empty patches when the decision text mentions stale code
  together with `must remove` or `undeclared`.
- Added a regression test for a stale-code MMA patch warning.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 53 passed.
- `uv run --extra dev pytest`: 142 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.
- Runtime push/fetch verification after code guard: local `main` and `origin/main` both resolved to
  `53b90824e7003831fa5e3e5bcafacca3837fad09`.
- Runtime push/fetch verification after knowledge update: local `main` and `origin/main` both
  resolved to `400aaa5ddd2e4c375494ca8df7902d527e9fd125`.

## 2026-05-08 - Checkpoint 3.70: Warp-row WMMA single-tile probe limits

Success criteria for this checkpoint:

- Run one bounded loop after the stale MMA patch guard.
- Capture whether the agent can produce a different Ampere direction after the supervisor reset.

Research note:

- Exa surfaced NVIDIA warp-level primitive guidance: participating threads must execute warp-level
  collectives coherently. This reinforces that WMMA work must be warp-wide, not single-lane.

Loop result:

- The agent proposed a compile-only warp-row WMMA single-tile probe.
- The patch added `mma.h`, WMMA fragment declarations, and a guarded QK `mma_sync` block.
- The patch applied and compiled successfully; cleanup reverted it afterward.

Compile diagnostics:

- BF16/Half stayed at 48 registers, 1 barrier, 16896 bytes shared memory, and no spills.
- FP32 stayed at 56 registers, 1 barrier, 33280 bytes shared memory, and no spills.

Decision:

- This probe is not scoreable.
- The generated code gated WMMA work behind `lane == 0`, did not store the WMMA score fragment into
  the existing `scores` array, and would be skipped on the current head_dim128 target because
  `can_stage_shared` is false.
- Future warp-row WMMA work must be warp-wide and must route produced score tiles into the existing
  online-softmax/output path before scoring.

Verification:

- Runtime knowledge update passed `git diff --check`.
- Runtime push/fetch verification: local `main` and `origin/main` both resolved to
  `f1e7a21b0f070d1de97057de70f41d4820ffab58`.

## 2026-05-08 - Checkpoint 3.71: Correctness-breaking patch self-critique guard

Success criteria for this checkpoint:

- Run one bounded loop after recording the warp-row WMMA probe.
- Prevent patches from executing when their own decision text says they will break correctness.

Research note:

- Exa refreshed NVIDIA's Ampere tuning guide. The relevant constraints remain unchanged for A6000:
  compile explicitly for `sm_86`, use Ampere async-copy/HMMA primitives rather than Blackwell
  TMA/WGMMA, and respect compute capability 8.6 occupancy/shared-memory limits.

Loop result:

- The agent proposed a tiled online-softmax rescale patch.
- The patch changed the correct invariant from
  `output_acc = output_acc * old_scale + tile_acc * tile_scale` to
  `output_acc = output_acc * old_scale + tile_acc`.
- The patch applied and compiled successfully on `sm_86`, with ptxas reporting no spills.
- Cleanup reverted the patch afterward because the command was compile-only and no lineage gate
  decision was made.

Reliability gap:

- The decision risk text explicitly said the original formula was correct, the patch would break
  correctness, and the direction should be rejected.
- Existing self-invalid guards caught `will cause a compile error` and stale-code warnings, but did
  not catch this correctness-breaking self-critique.

Runtime change:

- Planner validation now rejects non-empty patches when the decision text includes
  `will break correctness` or `reject this direction`.
- Added a regression test for the tiled-rescale self-critique pattern.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 54 passed.
- `uv run --extra dev pytest`: 143 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.
- Runtime push/fetch verification after code guard: local `main` and `origin/main` both resolved to
  `df7b8a5399b52e3308f8eb9272ec64de51df89c5`.

## 2026-05-08 - Checkpoint 3.72: Repeated tiled seed smoke guard

Success criteria for this checkpoint:

- Run one bounded loop after adding the correctness-breaking self-critique guard.
- Prevent the agent from repeating known no-patch tiled smoke scores as progress attempts.

Loop result:

- The agent ran a no-edit score of `candidates/cuda_tiled_attention_seed.py` at the already-known
  tiny shape: `seq_len=16`, `head_dim=16`, `total_tokens=16`, `num_heads=1`, BF16, both masks.
- The score passed correctness but was too small to matter for the current lineage:
  noncausal `2.1690319638268874e-05` TFLOPS, causal `1.0581573161156648e-05` TFLOPS, geomean
  `1.5149841720005358e-05` TFLOPS.
- The gate rejected it because the benchmark case signature differs from the current warp-row best.

Reliability gap:

- The validator capped unpatched tiled scores to the tiny validated shape, but still allowed that
  recorded no-patch smoke to repeat.
- The decision text also described a compile diagnostic while issuing a score command, reinforcing
  that no-edit diagnostics need tighter usefulness checks.

Runtime change:

- Planner validation now rejects unpatched repeats of the recorded tiny tiled smoke score.
- The prompt context now tells the agent that the tiled no-patch smoke is already recorded and should
  not be repeated.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 54 passed.
- `uv run --extra dev pytest`: 143 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.
- Runtime push/fetch verification after code guard: local `main` and `origin/main` both resolved to
  `2c8dfaefb5133c0455ec663e2d6a4cc80fdcf499`.

## 2026-05-08 - Checkpoint 3.73: CuTe online-softmax source refresh

Success criteria for this checkpoint:

- Refresh source-backed guidance before another tiled-kernel loop.
- Record the relevant online-softmax recurrence in runtime knowledge.

Research result:

- Exa surfaced Dao-AILab's CuTe FlashAttention online softmax helper at commit `58fe37fb`.
- The helper computes a row scale from the previous row max to the current row max, updates row sum
  with the prior row sum multiplied by that scale, and provides `rescale_O` to scale the accumulated
  output before consuming the current probability tile.

Decision:

- The local tiled-kernel invariant remains correct:
  `output_acc = output_acc * old_scale + tile_acc * tile_scale`, with
  `row_sum = row_sum * old_scale + tile_sum * tile_scale`.
- Future tiled fixes should diagnose indexing, synchronization, or bounds issues rather than
  removing `old_scale` or `tile_scale`.

Verification:

- Runtime knowledge update passed `git diff --check`.

## 2026-05-08 - Checkpoint 3.74: MMA pipeline skeleton compile failure guard

Success criteria for this checkpoint:

- Run one bounded loop after the CuTe online-softmax source refresh.
- Capture whether the agent can move away from repeated tiled no-patch attempts.
- Prevent repeat MMA pipeline patches with known invalid CUDA pipeline patterns.

Loop result:

- The agent proposed an MMA double-buffered async-copy skeleton for
  `candidates/cuda_mma_attention/attention_kernel.cu`.
- The patch applied, then `avo compile --source candidates/cuda_mma_attention/attention_kernel.cu
  --out-dir build/mma_double_buffer_skeleton` failed.
- Cleanup reverse-applied the patch successfully.

Compile failure:

- `__pipeline_wait_prior<1>()` failed because the local public CUDA primitive is
  `__pipeline_wait_prior(prior)`, not the templated spelling.
- The patch accidentally removed the opening `wmma::fragment<wmma::matrix_a, ...>` declaration line,
  leaving stray template arguments and an undefined `q_frag`.
- The patch also used scalar BF16 `__pipeline_memcpy_async(..., sizeof(__nv_bfloat16))` copies even
  though prior knowledge requires 16-byte aligned groups for useful Ampere async copies.

Runtime change:

- Planner validation now rejects candidate patches that add templated
  `__pipeline_wait_prior<...>`.
- Planner validation now rejects scalar BF16 `__pipeline_memcpy_async` additions and asks for
  16-byte aligned groups.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 56 passed.
- `uv run --extra dev pytest`: 145 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.
- Runtime push/fetch verification after code guard: local `main` and `origin/main` both resolved to
  `0aef1f375f481162c8262ec9b9d84e2c47b57f8c`.

## 2026-05-08 - Checkpoint 3.75: Official CUDA async-copy API refresh

Success criteria for this checkpoint:

- Cross-check the local pipeline-header finding against official NVIDIA docs.
- Record the actionable API spelling and alignment guidance before another loop.

Research result:

- Exa found NVIDIA's CUDA Programming Guide sections for advanced kernel programming and pipelines,
  plus the CCCL/libcu++ `cuda::memcpy_async` reference.
- The primitive API is documented as function-style `__pipeline_memcpy_async`,
  `__pipeline_commit`, and `__pipeline_wait_prior(N)`.
- The higher-level pipeline API uses `cuda::pipeline` with producer acquire, `cuda::memcpy_async`,
  producer commit, and consumer wait.
- On Ampere+, aligned global-to-shared `cuda::memcpy_async` can lower to `cp.async`.

Decision:

- Keep rejecting templated `__pipeline_wait_prior<...>` patches.
- Keep requiring 16-byte grouped async-copy candidates for BF16 tiles when the intent is useful
  Ampere `cp.async` staging rather than scalar copy noise.

Verification:

- Runtime knowledge update passed `git diff --check`.

## 2026-05-08 - Checkpoint 3.76: No-op async-copy stub guard

Success criteria for this checkpoint:

- Run one bounded loop after recording official CUDA async-copy API guidance.
- Prevent compile-only patches that add empty, uncalled helper stubs as false progress.

Loop result:

- The agent proposed a warp-row async-copy API proof patch.
- The patch added `cuda_pipeline_primitives.h`, small wrapper functions for
  `__pipeline_memcpy_async`, `__pipeline_commit`, and `__pipeline_wait_prior`, plus an empty
  `async_copy_tile_kv` helper.
- The patch applied, then `avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu
  --out-dir build/warp_async_copy_api_proof` failed.
- Cleanup reverse-applied the patch successfully.

Compile failure:

- The added helper used `scalar_t` outside a templated helper or kernel context.
- The compile output then cascaded into syntax errors around the existing score path.

Reliability gap:

- The decision risk text admitted the `async_copy_tile_kv` stub was empty, not called, and could not
  affect correctness or throughput.
- Such patches are compile-only noise because prior checkpoints already proved the async-copy API
  header and wrapper availability.

Runtime change:

- Planner validation now rejects non-empty patches whose own decision text says the patch is not yet
  called, has an empty stub, or cannot affect correctness or throughput.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 57 passed.
- `uv run --extra dev pytest`: 146 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.
- Runtime push/fetch verification after code guard: local `main` and `origin/main` both resolved to
  `177d640cc221f7f1b3f9df7873bb52263b6f4181`.

## 2026-05-08 - Checkpoint 3.77: Async-copy alignment nuance refresh

Success criteria for this checkpoint:

- Refresh Ampere async-copy guidance before the next loop.
- Preserve the useful 16-byte BF16 grouping rule without misstating the CUDA API minimum.

Research result:

- Exa re-surfaced the official CCCL/libcu++ `cuda::memcpy_async` reference and NVIDIA Ampere CUDA
  architecture material.
- CCCL documents that Ampere+ can lower aligned global-to-shared copies to `cp.async` with at least
  4-byte alignment.
- The Ampere architecture material calls out a better async-copy path when the data size and
  alignment are 16 bytes.

Decision:

- Keep rejecting scalar 2-byte BF16 `__pipeline_memcpy_async` patches as noise.
- Treat 4-byte aligned async copies as API-valid smokes or carefully justified tails, not the target
  for throughput work.
- Continue steering BF16 throughput attempts toward 16-byte groups, which correspond to eight BF16
  values, with scalar tails handled separately.

Verification:

- Runtime knowledge update passed `git diff --check`.

## 2026-05-08 - Checkpoint 3.78: Pragma-only compile guard

Success criteria for this checkpoint:

- Run one bounded loop after the async-copy alignment refresh.
- Prevent scoreable performance-only patches from stopping at compile-only diagnostics.

Loop result:

- The agent proposed a warp-row V-accumulation unroll patch.
- The patch added `#pragma unroll` before the `key_inner` V accumulation loops in both the shared
  staging path and the global fallback path.
- The patch applied, then `avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu
  --out-dir build/warp_v_accum_unroll` succeeded.
- Cleanup reverse-applied the patch successfully.
- No score payload was produced, so the lineage did not change.

Compile result:

- BF16 entry point: 48 registers, 1 barrier, 16896 bytes shared memory, no spills.
- FP16 entry point: 48 registers, 1 barrier, 16896 bytes shared memory, no spills.
- FP32 entry point: 56 registers, 1 barrier, 33280 bytes shared memory, no spills.

Reliability gap:

- The patch was correctness-neutral and explicitly targeted throughput, so a compile-only command
  could not determine whether the change improved or regressed TFLOPS.
- Compile-only checks remain useful for build-risk changes, but pragma-only or scheduler-only
  performance patches should run a bounded candidate score.

Runtime change:

- Planner validation now rejects compile commands when the candidate patch adds only `#pragma unroll`
  lines.
- The repo context now tells the agent to score pragma-only or scheduler-only performance patches
  instead of stopping at compile-only checks.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 58 passed.
- `uv run --extra dev pytest`: 147 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.

## 2026-05-08 - Checkpoint 3.79: Tiled wrapper-cap-only score guard

Success criteria for this checkpoint:

- Run one bounded loop with the pragma-only score guard in place.
- Prevent tiled larger-shape scores that only raise wrapper caps without fixing the kernel.

Loop result:

- The agent proposed a tiled reset attempt for `candidates/cuda_tiled_attention_seed.py`.
- The decision text claimed an online-softmax output rescaling fix, but the patch changed only
  `MAX_SMOKE_SEQUENCE` and `MAX_SMOKE_HEAD_DIM` wrapper caps.
- The patched candidate scored `seq_len=64`, `head_dim=64`, `total_tokens=256`, `num_heads=4`,
  BF16, both causal modes.
- Cleanup reverse-applied the wrapper patch successfully.
- The lineage did not change.

Score result:

- Noncausal failed correctness with `max_abs_error=0.6171875`, median `0.9199039936065674 ms`,
  and `0.0` TFLOPS.
- Causal failed correctness with `max_abs_error=0.2646484375`, median `0.7273600101470947 ms`,
  and `0.0` TFLOPS.
- Gate rejected the candidate because it failed correctness.

Reliability gap:

- The tiled kernel is only validated at the tiny `seq_len=16`, `head_dim=16`, `total_tokens=16`,
  `num_heads=1` smoke shape.
- Raising wrapper caps alone does not address the known larger-shape tiled correctness failure.

Runtime change:

- Planner validation now rejects tiled scores outside the tiny validated shape when the patch only
  changes `candidates/cuda_tiled_attention_seed.py` wrapper caps.
- The repo context now tells the agent that larger tiled scores need a kernel change, not only a
  wrapper-cap edit.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 59 passed.
- `uv run --extra dev pytest`: 148 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.

## 2026-05-08 - Checkpoint 3.80: WMMA skeleton whitespace and self-break guard

Success criteria for this checkpoint:

- Run one bounded loop after the tiled wrapper-cap-only guard.
- Prevent invalid structural proof patches from reaching `git apply --check` when the diff or
  decision text already exposes the failure.

Loop result:

- The agent proposed a warp-row WMMA QK scoring skeleton for `head_dim=128`.
- The patch attempted to add `<mma.h>`, WMMA fragment declarations, eight 16-wide K chunks, a
  `wmma::store_matrix_sync` score tile, and an early return that would keep the skeleton isolated
  from the scalar path.
- The patch was rejected before command execution because `git apply --check` failed.
- No compile or score payload was produced, and the lineage did not change.

Rejected patch diagnostics:

- `git apply --check` reported trailing whitespace in added lines and squelched additional
  whitespace errors.
- The decision risk text admitted the early return would break correctness if scored.

Runtime change:

- Planner validation now rejects candidate patches with trailing whitespace in added diff lines.
- Planner validation now treats `would break correctness` as self-invalid language for a non-empty
  patch, matching the existing `will break correctness` guard.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 61 passed.
- `uv run --extra dev pytest`: 150 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.

## 2026-05-08 - Checkpoint 3.81: WMMA BF16 source refresh

Success criteria for this checkpoint:

- Refresh source-backed WMMA guidance after another invalid warp-row WMMA skeleton.
- Record constraints that should shape the next tensor-core attempt.

Research result:

- Exa found NVIDIA CUTLASS functionality tables and NVIDIA's `bf16TensorCoreGemm` CUDA sample.
- CUTLASS documents SM80+ BF16 TensorOp support and lists WMMA 16x16x16 warp-level shapes.
- CUTLASS also notes that WMMA shared-memory loads expect 128-bit alignment.
- NVIDIA's BF16 sample stages A/B tiles through shared memory, uses 16-byte vectorized copies,
  adds BF16 skew/padding, and uses explicit `__nv_bfloat16` WMMA fragments.

Decision:

- Future warp-row WMMA work should avoid dead direct-global-load skeletons and early returns.
- A useful compile step should first build an aligned shared-memory tile path that can later feed
  the existing online-softmax score path.
- WMMA patches should preserve the scalar path until the tensor-core score tile is fully integrated
  and scoreable.

Verification:

- Runtime knowledge update passed `git diff --check`.

## 2026-05-08 - Checkpoint 3.82: Accepted warp-row shared-memory skew

Success criteria for this checkpoint:

- Run one bounded loop after the WMMA BF16 source refresh.
- Accept only a candidate that passes correctness and improves the current seq256/head_dim128 BF16
  warp-row lineage score.

Loop result:

- The agent proposed adding one padding column to both staged K and V shared-memory tiles:
  `k_tiles[kTileKeys][kMaxHeadDim + 1]` and `v_tiles[kTileKeys][kMaxHeadDim + 1]`.
- The hypothesis was that the original stride of 128 BF16 elements caused shared-memory bank
  conflicts during V accumulation, and that a one-column skew could reduce those conflicts.
- The patch applied and scored the existing warp-row BF16 suite at `seq_len=256`, `head_dim=128`,
  `total_tokens=1024`, `num_heads=4`, both causal modes.
- The candidate passed correctness and the throughput gate.
- Nested lineage accepted commit: `259be1d619c58eb1c52faf6fe8e57b16b2121b56`.

Score result:

- Previous best geomean: `0.4012802607933843` TFLOPS.
- Candidate geomean: `0.43185073056556733` TFLOPS.
- Noncausal: correct, `max_abs_error=0.001953125`, median `0.8902720212936401 ms`,
  `0.60304142909027` TFLOPS.
- Causal: correct, `max_abs_error=0.015625`, median `0.8679999709129333 ms`,
  `0.3092574481513733` TFLOPS.

Compile verification:

- `avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu
  --out-dir build/warp_shared_skew_accept_verify` succeeded for `sm_86`.
- BF16 entry point: 48 registers, 1 barrier, 17024 bytes shared memory, no spills.
- FP16 entry point: 48 registers, 1 barrier, 17024 bytes shared memory, no spills.
- FP32 entry point: 56 registers, 1 barrier, 33536 bytes shared memory, no spills.

Runtime change:

- Runtime candidate source now carries the accepted shared-memory skew patch.
- Runtime knowledge records the new accepted best and ptxas resource counts.

Verification:

- `uv run --extra dev pytest`: 150 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.

## 2026-05-08 - Checkpoint 3.83: Pragma-only scored regression guard

Success criteria for this checkpoint:

- Run one bounded loop from the accepted shared-memory skew best.
- Record whether a standalone unroll pragma can build on the new best.
- Prevent repeat standalone pragma-only patches if the scored result regresses.

Loop result:

- The agent proposed adding `#pragma unroll` before the two V accumulation `key_inner` loops in the
  warp-row kernel.
- The patch applied and scored the existing seq256/head_dim128 BF16 warp-row suite.
- The candidate passed correctness but regressed geomean throughput.
- Cleanup reverse-applied the patch successfully, so the runtime source kept only the accepted
  shared-memory skew.
- The lineage did not change.

Score result:

- Current best geomean: `0.43185073056556733` TFLOPS.
- Candidate geomean: `0.2576601941393183` TFLOPS.
- Noncausal: correct, `max_abs_error=0.001953125`, median `1.6269760131835938 ms`,
  `0.32998084031335845` TFLOPS.
- Causal: correct, `max_abs_error=0.015625`, median `1.3342399597167969 ms`,
  `0.20118978902189197` TFLOPS.
- Gate rejected the candidate because it regressed geomean throughput.

Runtime change:

- Planner validation now rejects standalone pragma-only performance patches.
- Unroll pragmas remain possible only when paired with a substantive code change and a bounded score.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 62 passed.
- `uv run --extra dev pytest`: 151 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.

## 2026-05-08 - Checkpoint 3.84: Dynamic shared-memory skeleton self-invalid guard

Success criteria for this checkpoint:

- Run one bounded loop after the standalone pragma-only guard.
- Prevent structural compile-only patches whose own risk text says they cannot improve throughput
  and need more indexing work before scoring.

Loop result:

- The agent proposed a dynamic shared-memory double-buffer skeleton for future warp-row `cp.async`.
- The patch moved the skewed K/V tiles from static 2D arrays into flat `extern __shared__` buffers,
  added dtype-specific dynamic shared-memory sizing, and set a max dynamic shared-memory attribute.
- The patch applied, then `avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu
  --out-dir build/warp_dyn_shared_skeleton` failed.
- Cleanup reverse-applied the patch successfully.
- No score payload was produced, and the lineage did not change.

Compile failure:

- The patch left existing 2D access sites such as `k_tiles[lane][0]`,
  `v_tiles[key_inner][dim]`, and staged stores unchanged after converting `k_tiles` and `v_tiles`
  to flat `scalar_t*` pointers.
- NVCC reported `no operator "[]" matches these operands` and failed template instantiation.

Reliability gap:

- The risk text admitted the doubled buffers were unused.
- It also said the patch could not improve throughput and that the existing indexing had to be
  updated before scoring.

Runtime change:

- Planner validation now treats `cannot improve throughput`, `unused doubled buffers`, and
  `must be updated before scoring` as self-invalid language for non-empty patches.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 63 passed.
- `uv run --extra dev pytest`: 152 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.

## 2026-05-08 - Checkpoint 3.85: Dynamic K/V flat-buffer compile proof

Success criteria for this checkpoint:

- Run one bounded loop after the dynamic shared-memory self-invalid guard.
- Determine whether a corrected dynamic K/V shared-memory migration can at least compile.

Loop result:

- The agent proposed moving the warp-row K/V staging tiles from static 2D shared arrays into flat
  `extern __shared__` buffers.
- The patch kept `score_tiles` static, introduced `kv_stride = kMaxHeadDim + 1`, replaced 2D K/V
  accesses with flat `key * kv_stride + dim` indexing, and passed dtype-specific dynamic
  shared-memory byte counts at launch.
- The patch applied, then `avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu
  --out-dir build/warp_dyn_kv_tiles` succeeded.
- Cleanup reverse-applied the patch because it was compile-only.
- No score payload was produced, and the lineage did not change.

Compile result:

- BF16 entry point: 48 registers, 1 barrier, 512 bytes static shared memory, no spills.
- FP16 entry point: 48 registers, 1 barrier, 512 bytes static shared memory, no spills.
- FP32 entry point: 56 registers, 1 barrier, 512 bytes static shared memory, no spills.

Decision:

- The dynamic K/V flat-buffer migration is a valid compile proof.
- Before adding double buffering or `cp.async`, the same migration should be scored on the current
  seq256/head_dim128 BF16 suite to prove correctness and throughput.

Verification:

- Runtime knowledge update passed `git diff --check`.

## 2026-05-08 - Checkpoint 3.86: Dynamic K/V migration score regression guard

Success criteria for this checkpoint:

- Run one bounded loop after recording the dynamic K/V flat-buffer compile proof.
- Score the dynamic K/V migration before adding `cp.async` or double buffering.
- Prevent repeating the dynamic-shared migration if it regresses as a standalone change.

Loop result:

- The agent proposed applying and scoring the dynamic K/V shared-memory migration.
- The patch moved K/V tiles into flat `extern __shared__` buffers, converted K/V accesses to flat
  offset indexing, passed a dynamic shared-memory byte count at launch, and attempted a dynamic
  shared-memory attribute when needed.
- The patch applied and scored the existing seq256/head_dim128 BF16 warp-row suite.
- The candidate passed correctness but regressed geomean throughput.
- Cleanup reverse-applied the patch successfully, and the lineage did not change.

Score result:

- Current best geomean: `0.43185073056556733` TFLOPS.
- Candidate geomean: `0.32420366797036887` TFLOPS.
- Noncausal: correct, `max_abs_error=0.001953125`, median `1.2728320360183716 ms`,
  `0.42179242571503833` TFLOPS.
- Causal: correct, `max_abs_error=0.015625`, median `1.0772160291671753 ms`,
  `0.24919370741961078` TFLOPS.
- Gate rejected the candidate because it regressed geomean throughput.

Runtime change:

- Planner validation now rejects standalone dynamic shared-memory K/V migrations that do not also
  add real async-copy or double-buffering logic.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 64 passed.
- `uv run --extra dev pytest`: 153 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.

## 2026-05-08 - Checkpoint 3.87: Head-dim128 shared threshold guard

Success criteria for this checkpoint:

- Run one bounded loop after the dynamic K/V migration regression guard.
- Evaluate the proposed direct `head_dim <= 128` shared-path threshold change.
- Prevent repeating the threshold-only change if it is unsafe.

Loop result:

- The agent proposed changing `can_stage_shared` from `head_dim <= 64` to `head_dim <= 128`.
- The generated patch was rejected before scoring because `git apply --check` reported a corrupt
  patch.
- The underlying one-line change was simple, so it was applied manually and scored with the same
  seq256/head_dim128 BF16 suite.
- The manual score failed correctness, so the threshold change was reverted.
- The lineage did not change.

Manual score result:

- Noncausal failed with a CUDA unknown error before timing samples were recorded.
- Causal failed with a CUDA misaligned-address error before timing samples were recorded.
- Candidate geomean was `0.0` TFLOPS.

Runtime change:

- Planner validation now rejects direct threshold-only patches that set the warp-row shared path to
  `head_dim <= 128`.
- Future head_dim128 shared-path attempts must include a real alignment or layout fix, not only a
  threshold increase.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 65 passed.
- `uv run --extra dev pytest`: 154 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.

## 2026-05-08 - Checkpoint 3.88: Head-dim128 shared alignment diagnosis

Success criteria for this checkpoint:

- Research the likely cause of the head_dim128 shared-path misaligned-address failure.
- Record the local alignment invariant future patches must preserve.

Research result:

- Exa surfaced NVIDIA Ampere shared-memory limits, CUTLASS SM80 async-copy code, and cuBLASDx
  performance guidance.
- The useful recurring point is that vectorized/shared-memory paths should preserve alignment,
  with NVIDIA/CUTLASS material repeatedly emphasizing 16-byte alignment for efficient copies and
  shared-memory IO.

Local diagnosis:

- The warp-row `dot_product` uses `ScalarPack4` reinterpret loads when `head_dim` is divisible by
  four.
- The accepted K/V skew uses `kMaxHeadDim + 1`, so BF16 shared rows have a 129-element stride, or
  258 bytes.
- That stride does not preserve `ScalarPack4` alignment across rows, and it is also not 16-byte
  aligned.
- This explains why simply enabling the shared path at head_dim128 caused CUDA unknown and
  misaligned-address failures.

Decision:

- Future head_dim128 shared-path attempts must fix row-stride alignment or use a scalar shared-row
  dot path before raising `can_stage_shared`.
- A safe direction is a stride that is a multiple of four BF16 elements for the current
  `ScalarPack4`, or a multiple of eight BF16 elements when targeting 16-byte alignment.

Verification:

- Runtime knowledge update passed `git diff --check`.

## 2026-05-08 - Checkpoint 3.89: MMA score-fragment K32 guard

Success criteria for this checkpoint:

- Run one bounded loop after the head_dim128 shared-path alignment diagnosis.
- Record the next MMA head_dim32 compile failure precisely enough to avoid repeating it.
- Add a narrow validator guard that does not block valid widened PV/output fragments.

Loop result:

- The agent switched to the MMA seed and proposed a compile-first head_dim32 two-chunk WMMA patch.
- The patch updated `SMOKE_HEAD_DIM` and `kHeadDim` to 32, widened `pv_tile` and `output_acc` to
  `kTile * kHeadDim`, removed the stale single-chunk QK code, and used two 16-wide chunks for QK
  and PV.
- The patch applied, then `avo compile --source candidates/cuda_mma_attention/attention_kernel.cu
  --out-dir build/mma_head_dim_32_two_chunk` failed.
- Cleanup reverse-applied the patch successfully, and the lineage did not change.

Compile failure:

- NVCC rejected `wmma::fragment<wmma::accumulator, kTile, kTile, kHeadDim, float>` after
  `kHeadDim = 32`.
- The instantiated shape was `fragment<accumulator, 16, 16, 32, float>`, which is incomplete or
  unsupported for this WMMA score accumulator.
- The correct direction remains two 16-wide QK fragments that accumulate into a valid 16x16 score
  accumulator, not a score accumulator whose K dimension is 32.

Runtime change:

- Planner validation now rejects candidate patches that introduce the unsupported symbolic
  `kTile,kTile,kHeadDim` score accumulator after setting `kHeadDim = 32`.
- It also rejects the literal `fragment<accumulator, 16, 16, 32, float>` form.
- The guard intentionally does not reject valid PV/output accumulator shapes such as
  `kTile,16,kTile`.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 67 passed.
- `uv run --extra dev pytest`: 156 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.

## 2026-05-08 - Checkpoint 3.90: WMMA scalar_t fragment guard

Success criteria for this checkpoint:

- Run one bounded loop after the MMA score-fragment guard.
- Record the next warp-row WMMA compile failure precisely enough to avoid repeating it.
- Turn the existing explicit-WMMA-type knowledge into a validator rule.

Source refresh:

- Exa surfaced NVIDIA/CUTLASS material for Ampere BF16 Tensor Core and WMMA shapes.
- The relevant source-backed direction remains to use explicit CUDA WMMA element types and valid
  Ampere shapes, not generic PyTorch `scalar_t` fragments.

Loop result:

- The agent proposed adding a minimal warp-row WMMA QK skeleton inside the shared K tile loop.
- The patch added `mma.h`, declared 16x16x16 matrix A/B fragments using `scalar_t`, and attempted
  to store one WMMA score tile while leaving the scalar path intact.
- The patch applied, then `avo compile --source candidates/cuda_warp_rows_attention/attention_kernel.cu
  --out-dir build/warp_wmma_qk_skeleton` failed.
- Cleanup reverse-applied the patch successfully, and the lineage did not change.

Compile failure:

- NVCC instantiated the templated warp-row kernel for `float`, `c10::Half`, and `c10::BFloat16`.
- All `wmma::fragment<wmma::matrix_a|matrix_b, 16, 16, 16, scalar_t, ...>` instantiations were
  rejected as incomplete or unsupported.
- Future warp-row WMMA patches need explicit CUDA fragment element types, such as
  `__nv_bfloat16`, in dtype-specific code paths.

Runtime change:

- Planner validation now rejects candidate patches that use `scalar_t` as a WMMA matrix fragment
  element.
- The repo context prompt also states that generic PyTorch WMMA fragments must use explicit CUDA
  element types, because dispatch instantiates unsupported float/c10 types.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 68 passed.
- `uv run --extra dev pytest`: 157 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.

## 2026-05-08 - Checkpoint 3.91: Stale tiled rescale guard

Success criteria for this checkpoint:

- Run one bounded loop after the WMMA scalar_t fragment guard.
- Decide whether the proposed tiled rescale fix still applies to the current source.
- Prevent repeating the stale tiled rescale patch if the current kernel already contains it.

Loop result:

- The agent proposed fixing tiled online-softmax output accumulation by changing
  `output_acc = tile_acc * tile_scale` to
  `output_acc = output_acc * old_scale + tile_acc * tile_scale`.
- The generated patch was rejected before scoring because `git apply --check` reported a corrupt
  patch.
- Manual inspection showed the current tiled source already contains the correct recurrence, so
  there was no source change to apply or score.
- The lineage did not change.

Runtime change:

- Planner validation now rejects candidate patches that repeat that stale tiled rescale replacement.
- The repo context prompt also states that the current tiled kernel already has the correct
  online-softmax output recurrence.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 69 passed.
- `uv run --extra dev pytest`: 158 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.

## 2026-05-08 - Checkpoint 3.92: MMA incomplete-removal guard

Success criteria for this checkpoint:

- Run one bounded loop after the stale tiled rescale guard.
- Record the next generated MMA head_dim32 patch failure.
- Reject patches whose own risk text identifies incomplete removal of old code.

Loop result:

- The agent proposed another compile-first MMA head_dim32 two-chunk patch.
- The patch kept score WMMA K at 16 and used explicit `__nv_bfloat16`, but still left old
  single-chunk PV lines after the new two-chunk PV loop.
- `git apply --check` rejected the patch as corrupt before compile.
- The lineage did not change.

Decision:

- The patch was self-invalid: its risk text said incomplete removal of old single-chunk lines was
  the main risk and those old lines should be completely removed.
- Planner validation now rejects non-empty patches with that incomplete-removal warning, forcing a
  corrected diff before `git apply`.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 70 passed.
- `uv run --extra dev pytest`: 159 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.

## 2026-05-08 - Checkpoint 3.93: MMA orphan k_frag guard

Success criteria for this checkpoint:

- Run one bounded loop after the incomplete-removal guard.
- Record the next generated MMA head_dim32 patch failure.
- Add a narrow guard for the new stale-fragment pattern.

Loop result:

- The agent proposed another compile-first MMA head_dim32 two-chunk patch.
- The patch used valid 16-wide WMMA chunks and explicit `__nv_bfloat16`, but inserted an orphan
  post-QK `wmma::fragment<wmma::matrix_b, ...> k_frag;` block after storing the score tile.
- `git apply --check` rejected the patch as corrupt before compile.
- The lineage did not change.

Runtime change:

- Planner validation now rejects the exact pattern where a patch stores `score_frag`, synchronizes,
  and then leaves a standalone post-QK `k_frag` block.
- A valid two-chunk QK rewrite must remove all old single-chunk fragment declarations.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 71 passed.
- `uv run --extra dev pytest`: 160 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.

## 2026-05-08 - Checkpoint 3.94: Manual MMA head_dim32 compile/score proof

Success criteria for this checkpoint:

- Manually complete the repeated MMA head_dim32 two-chunk direction after generated patches failed
  on stale fragments and invalid WMMA score shapes.
- Verify that the CUDA seed compiles and scores correctly on the tiny head_dim32 smoke shape.
- Align the agent validator, repo context, and tests with the new committed MMA seed shape.

Runtime change:

- The MMA seed now uses `kHeadDim = 32`.
- QK keeps a valid 16x16x16 score accumulator and accumulates two 16-wide chunks with explicit
  `__nv_bfloat16` WMMA matrix fragments.
- PV also runs two 16-wide chunks, stores each chunk at the row-major column offset
  `&pv_tile[chunk * 16]`, and uses a 32-wide leading dimension.
- `pv_tile` and `output_acc` are widened to 16x32 while `scores` and `probabilities` remain 16x16.
- The wrapper and planner now advertise the unpatched MMA smoke cap as seq_len 16/32,
  head_dim 32, total_tokens <= 32, and num_heads 1.
- Patched MMA score attempts beyond the current head_dim32 smoke still require an `avo compile`
  build-check before scoring.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 72 passed.
- `uv run --extra dev pytest`: 161 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.
- `uv run --extra cuda python -m avo compile --source candidates/cuda_mma_attention/attention_kernel.cu
  --out-dir build/manual_mma_head_dim32`: passed on sm86 with no spills, 40 registers, 1 barrier,
  5824 bytes shared memory, 400 bytes `cmem[0]`, 224 bytes `cmem[4]`, and 28 bytes global memory.
- `uv run --extra cuda python -m avo score --backend candidate --candidate
  candidates/cuda_mma_attention_seed.py --seq-lens 32 --total-tokens 32 --num-heads 1
  --head-dim 32 --dtype bf16 --causal both --repeats 1 --warmup 1 --trials 3 --timeout-s 300`:
  all_correct true, geomean `0.0001245601243057133` TFLOPS.

Score details:

- Noncausal: correct, max_abs_error `0.00390625`, median `0.74099200963974 ms`,
  `0.00017688719756063954` TFLOPS, samples
  `[0.74099200963974, 0.6902719736099243, 0.8248639702796936]`.
- Causal: correct, max_abs_error `0.00390625`, median `0.7471680045127869 ms`,
  `8.771253533900277e-05` TFLOPS, samples
  `[0.7471680045127869, 0.6773759722709656, 0.7756479978561401]`.

Decision:

- This is structural correctness progress, not a lineage improvement. The workload signature is
  seq32/head_dim32 and is not comparable to the current seq256/head_dim128 warp-row best.

## 2026-05-08 - Checkpoint 3.95: BF16 score_tiles regression guard

Success criteria for this checkpoint:

- Run one bounded loop after the committed MMA head_dim32 proof.
- Record the next warp-row attempt and its gate decision.
- Add a narrow guard for the exact regressed patch pattern if the attempt is not useful to repeat.

Loop result:

- The agent proposed converting the warp-row shared `score_tiles` buffer from FP32 to BF16.
- The patch changed `__shared__ float score_tiles[kRowsPerBlock][kTileKeys]` to
  `__shared__ __nv_bfloat16 score_tiles[kRowsPerBlock][kTileKeys]`, changed the local `scores`
  pointer to BF16, stored shifted probabilities through `__float2bfloat16`, and converted them
  back with `__bfloat162float` during V accumulation.
- The patch applied and scored successfully, then the lineage gate rejected it for throughput
  regression. Cleanup reverse-applied the patch successfully, so no source change leaked into the
  worktree.

Score result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --attempts-dir benchmarks/attempts --loop-json
  benchmarks/latest-loop.json --max-steps 1 --timeout-s 300 --env-file ../avo/.env.local`.
- Candidate score command: `avo score --backend candidate --candidate
  candidates/cuda_warp_rows_attention_seed.py --seq-lens 256 --total-tokens 1024 --num-heads 4
  --head-dim 128 --dtype bf16 --causal both --repeats 1 --warmup 1 --trials 3 --timeout-s 300`.
- All correctness checks passed.
- Noncausal: max_abs_error `0.001953125`, median `1.2097599506378174 ms`,
  `0.44378300977557367` TFLOPS, samples
  `[1.3521599769592285, 1.2097599506378174, 1.1501760482788086]`.
- Causal: max_abs_error `0.00390625`, median `0.937279999256134 ms`,
  `0.28639836144273007` TFLOPS, samples
  `[0.8016960024833679, 0.937279999256134, 1.0807360410690308]`.
- Geomean: `0.3565090838055145` TFLOPS.
- Gate decision: rejected versus best geomean `0.43185073056556733` because the candidate
  regressed throughput.

Runtime change:

- Planner validation now rejects the exact warp-row BF16 `score_tiles` conversion pattern.
- The repo context prompt also tells agents not to repeat that buffer-precision change.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 73 passed.
- `uv run --extra dev pytest`: 162 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.

## 2026-05-08 - Checkpoint 3.96: Manual MMA head_dim64 compile/score proof

Success criteria for this checkpoint:

- Run one bounded loop after the BF16 `score_tiles` guard.
- Preserve the useful compile-clean part of the generated MMA head_dim64 direction.
- Verify the manually applied four-chunk MMA seed with compile and score checks.

Loop result:

- The agent proposed extending the MMA seed from head_dim32 to head_dim64 by changing
  `kHeadDim` to 64 and changing the QK/PV chunk loops from two 16-wide chunks to four.
- The patch applied and `avo compile --source candidates/cuda_mma_attention/attention_kernel.cu
  --out-dir build/mma_head_dim_64_four_chunk` passed on sm86.
- The orchestrator cleaned the patch up because the loop step was compile-only and produced no
  score/gate decision.

Manual runtime change:

- Reapplied the four-chunk head_dim64 structure and fixed the stale runtime error message.
- The MMA seed now uses `kHeadDim = 64`, keeps score/probability tiles at 16x16, widens
  `pv_tile` and `output_acc` to 16x64, and stores each PV chunk at `&pv_tile[chunk * 16]`
  with leading dimension 64.
- The wrapper, README smoke command, agent repo context, score validator, and tests now treat
  seq_len 16/32, head_dim 64, total_tokens <= 32, and num_heads 1 as the unpatched MMA cap.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 73 passed.
- `uv run --extra cuda python -m avo compile --source candidates/cuda_mma_attention/attention_kernel.cu
  --out-dir build/manual_mma_head_dim64`: passed on sm86 with no spills, 40 registers, 1 barrier,
  9920 bytes shared memory, 400 bytes `cmem[0]`, 224 bytes `cmem[4]`, and 28 bytes global memory.
- `uv run --extra cuda python -m avo score --backend candidate --candidate
  candidates/cuda_mma_attention_seed.py --seq-lens 32 --total-tokens 32 --num-heads 1
  --head-dim 64 --dtype bf16 --causal both --repeats 1 --warmup 1 --trials 3 --timeout-s 300`:
  all_correct true, geomean `0.0003318536197406504` TFLOPS.
- `uv run --extra dev pytest`: 162 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.

Score details:

- Noncausal: correct, max_abs_error `0.00390625`, median `0.6331200003623962 ms`,
  `0.00041405104853732224` TFLOPS, samples
  `[0.6331200003623962, 0.5443199872970581, 0.8219199776649475]`.
- Causal: correct, max_abs_error `0.00390625`, median `0.4927999973297119 ms`,
  `0.0002659740274152339` TFLOPS, samples
  `[0.49248000979423523, 0.4991680085659027, 0.4927999973297119]`.

Decision:

- This is structural correctness progress, not a lineage improvement. The workload signature is
  seq32/head_dim64 and remains incomparable to the current seq256/head_dim128 warp-row best.

## 2026-05-08 - Checkpoint 3.97: Tiled reduction-bound non-fix guard

Success criteria for this checkpoint:

- Run one bounded loop after the MMA head_dim64 proof.
- Record whether the proposed tiled larger-shape correctness fix works.
- Reject the exact patch pattern if it does not fix correctness.

Loop result:

- The agent switched to the tiled seed and proposed guarding reduction lanes outside `tile_keys`.
- The patch initialized out-of-tile max-reduction lanes to `-inf`, out-of-tile sum-reduction lanes
  to `0.0f`, and wrote valid lanes through `reduce[tid] = score` and `reduce[tid] = shifted`.
- The patch applied and scored, but both causal modes still failed tolerance on
  seq128/head_dim128 BF16.
- Cleanup reverse-applied the patch successfully, so no source change leaked into the worktree.

Score result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --attempts-dir benchmarks/attempts --loop-json
  benchmarks/latest-loop.json --max-steps 1 --timeout-s 300 --env-file ../avo/.env.local`.
- Candidate score command: `avo score --backend candidate --candidate
  candidates/cuda_tiled_attention_seed.py --seq-lens 128 --total-tokens 512 --num-heads 4
  --head-dim 128 --dtype bf16 --causal both --repeats 1 --warmup 1 --trials 3 --timeout-s 300`.
- Noncausal: failed correctness, max_abs_error `0.672119140625`, median
  `1.0005120038986206 ms`, samples
  `[1.132383942604065, 1.0005120038986206, 0.8937600255012512]`.
- Causal: failed correctness, max_abs_error `0.927734375`, median
  `0.7628480195999146 ms`, samples
  `[0.97980797290802, 0.7352319955825806, 0.7628480195999146]`.
- Geomean: `0.0` TFLOPS.
- Gate decision: rejected versus best geomean `0.43185073056556733` because the candidate failed
  correctness.

Runtime change:

- Planner validation now rejects the exact tiled reduction-bound guard pattern because it does not
  fix the seq128/head_dim128 correctness failure.
- The repo context prompt also tells agents not to repeat that `reduce[tid]` score/shifted guard
  change.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 74 passed.
- `uv run --extra dev pytest`: 163 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.

## 2026-05-08 - Checkpoint 3.98: Partial MMA head_dim128 guard

Success criteria for this checkpoint:

- Run one bounded loop after the tiled reduction-bound non-fix guard.
- Record whether the next MMA head_dim128 step is scoreable or self-invalid.
- Add a validator guard if the patch admits it would fail correctness.

Source refresh:

- Exa refreshed official NVIDIA CUDA guidance before the loop: shared memory has 32 banks,
  bank conflicts serialize same-bank warp accesses, padding is the documented fix for strided
  bank conflicts, and Ampere asynchronous copies require careful synchronization and benefit from
  aligned global/shared memory.

Loop result:

- The agent proposed extending the MMA seed from head_dim64 to head_dim128.
- The patch changed `kHeadDim` and `SMOKE_HEAD_DIM` from 64 to 128 and added `#pragma unroll`
  before the existing QK/PV chunk loops.
- The patch applied and compiled, then was cleaned up because the step was compile-only.
- The patch was self-invalid: its risk text said the four 16-wide chunks still cover only 64
  dimensions and that scoring without a second pass would fail correctness.

Compile result:

- Command: `avo compile --source candidates/cuda_mma_attention/attention_kernel.cu
  --out-dir build/mma_head_dim_128_single_pass`.
- Compile passed on sm86 with no spills, 40 registers, 1 barrier, 18112 bytes shared memory,
  400 bytes `cmem[0]`, 224 bytes `cmem[4]`, and 28 bytes global memory.
- No score was run and no lineage gate decision was made.

Runtime change:

- Planner validation now rejects patches whose decision text says they will fail correctness.
- It also rejects the exact partial MMA head_dim128 pattern: changing `kHeadDim` and
  `SMOKE_HEAD_DIM` to 128 without adding a wider chunk loop or head-chunk abstraction.
- The repo context prompt tells agents not to repeat the partial head_dim128 extension.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 76 passed.
- `uv run --extra dev pytest`: 165 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed.

## 2026-05-08 - Checkpoint 3.99: Manual MMA head_dim128 compile/score proof

Success criteria for this checkpoint:

- Run one bounded loop after the partial MMA head_dim128 guard.
- Preserve the useful eight-chunk MMA head_dim128 direction after the orchestrator cleans up the
  compile-only patch.
- Verify the manually applied head_dim128 MMA seed with compile and score checks.

Loop result:

- The agent proposed extending the MMA seed from head_dim64 to head_dim128 by changing
  `kHeadDim` to 128 and changing the QK/PV chunk loops from four 16-wide chunks to eight.
- The patch applied and `avo compile --source candidates/cuda_mma_attention/attention_kernel.cu
  --out-dir build/mma_head_dim_128_eight_chunk` passed on sm86.
- The orchestrator cleaned the patch up because the loop step was compile-only and produced no
  score/gate decision.

Manual runtime change:

- Reapplied the eight-chunk head_dim128 structure.
- The MMA seed now uses `kHeadDim = 128`, keeps score/probability tiles at 16x16, widens
  `pv_tile` and `output_acc` to 16x128, and stores each PV chunk at `&pv_tile[chunk * 16]`
  with leading dimension 128.
- The wrapper, README smoke command, agent repo context, score validator, and tests now treat
  seq_len 16/32, head_dim 128, total_tokens <= 32, and num_heads 1 as the unpatched MMA cap.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: 76 passed.
- `uv run --extra cuda python -m avo compile --source candidates/cuda_mma_attention/attention_kernel.cu
  --out-dir build/manual_mma_head_dim128`: passed on sm86 with no spills, 40 registers, 1 barrier,
  18112 bytes shared memory, 400 bytes `cmem[0]`, 224 bytes `cmem[4]`, and 28 bytes global memory.
- `uv run --extra cuda python -m avo score --backend candidate --candidate
  candidates/cuda_mma_attention_seed.py --seq-lens 32 --total-tokens 32 --num-heads 1
  --head-dim 128 --dtype bf16 --causal both --repeats 1 --warmup 1 --trials 3 --timeout-s 300`:
  all_correct true, geomean `0.00038255968486606857` TFLOPS.
- `uv run --extra dev pytest`: 165 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

Score details:

- Noncausal: correct, max_abs_error `0.00390625`, median `0.9678720235824585 ms`,
  `0.0005416914501355385` TFLOPS, samples
  `[0.9489920139312744, 0.9950720071792603, 0.9678720235824585]`.
- Causal: correct, max_abs_error `0.001953125`, median `0.9702720046043396 ms`,
  `0.0002701757844769497` TFLOPS, samples
  `[0.9393280148506165, 1.143455982208252, 0.9702720046043396]`.

Decision:

- This is structural correctness progress, not a lineage improvement. The workload signature is
  seq32/head_dim128 and remains incomparable to the current seq256/head_dim128 warp-row best.

## 2026-05-08 - Checkpoint 4.00: Manual MMA seq64/head_dim128 proof

Success criteria for this checkpoint:

- Run one bounded loop after the manual MMA head_dim128 proof.
- Preserve a correctness-proven seq64 extension if it passes both causal modes.
- Align the planner context, score validator, README, and knowledge notes with the new cap.

Source refresh:

- Exa refreshed Ampere FlashAttention/CUTLASS context before the loop: NVIDIA's CUTLASS
  CuTeDSL Ampere FlashAttention v2 example combines `cp.async`, Ampere tensor-core MMA,
  register pipelining, online softmax, and output rescaling, while FlashAttention-2's sm8x
  head_dim128 heuristic uses smaller N-blocks on sm86. This supports continuing with small,
  correctness-proven tile extensions before adding async-copy overlap.

Loop result:

- The agent proposed extending the MMA seed from seq_len 16/32 to seq_len 64 while keeping
  head_dim 128 and the eight 16-wide QK/PV WMMA chunks.
- The patch changed `SMOKE_SEQUENCES` to `{16, 32, 64}`, changed `kMaxSeqLen` to 64, and scored
  `seq_len=64`, `total_tokens=256`, `num_heads=4`, `head_dim=128`, BF16, both causal modes.
- The patched score passed correctness but the lineage gate rejected it because the case signature
  differs from the current seq256/head_dim128 warp-row best.
- Cleanup reverse-applied the patch successfully, so no source change leaked from the orchestrator.

Manual runtime change:

- Reapplied the seq64 cap extension and verified it from the current worktree.
- The MMA seed now accepts seq_len 16/32/64, head_dim 128, total_tokens up to 256, and num_heads up
  to 4 as the validated smoke envelope.
- The repo context notes that the exact no-patch seq64 score is already recorded and should not be
  repeated without a new candidate patch.

Verification:

- `uv run --extra cuda python -m avo compile --source candidates/cuda_mma_attention/attention_kernel.cu
  --out-dir build/manual_mma_seq64_head_dim128`: passed on sm86 with no spills, 40 registers,
  1 barrier, 18112 bytes shared memory, 400 bytes `cmem[0]`, 224 bytes `cmem[4]`, and
  28 bytes global memory.
- `uv run --extra cuda python -m avo score --backend candidate --candidate
  candidates/cuda_mma_attention_seed.py --seq-lens 64 --total-tokens 256 --num-heads 4
  --head-dim 128 --dtype bf16 --causal both --repeats 1 --warmup 1 --trials 3 --timeout-s 300`:
  all_correct true, geomean `0.03636815047603261` TFLOPS.
- `uv run --extra dev pytest tests/test_agent.py -q`: 76 passed.
- `uv run --extra dev pytest`: 165 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

Score details:

- Noncausal: correct, max_abs_error `0.00390625`, median `0.6896960139274597 ms`,
  `0.04865104527562075` TFLOPS, samples
  `[0.5594239830970764, 0.6896960139274597, 0.7299839854240417]`.
- Causal: correct, max_abs_error `0.0078125`, median `0.6171200275421143 ms`,
  `0.027186309390769315` TFLOPS, samples
  `[0.6171200275421143, 0.6223679780960083, 0.5373119711875916]`.

Decision:

- This is structural correctness progress, not a lineage improvement. The workload signature is
  seq64/head_dim128 and remains incomparable to the current seq256/head_dim128 warp-row best.

## 2026-05-08 - Checkpoint 4.01: Manual MMA seq128/head_dim128 proof

Success criteria for this checkpoint:

- Run one bounded loop after the seq64 MMA proof.
- Preserve a compile-clean seq128 extension if it scores correctly.
- Align planner caps and notes so future work targets seq256 or structural kernel changes rather
  than repeating the no-patch seq128 score.

Loop result:

- The agent proposed extending the MMA seed from seq_len 64 to seq_len 128 while keeping
  head_dim 128 and the eight 16-wide QK/PV WMMA chunks.
- The patch changed `kMaxSeqLen` to 128 and changed `SMOKE_SEQUENCES` to
  `{16, 32, 64, 128}`.
- The patch applied and compiled, then was cleaned up because the loop step was compile-only.

Manual runtime change:

- Reapplied the seq128 cap extension and verified it from the current worktree.
- The MMA seed now accepts seq_len 16/32/64/128, head_dim 128, total_tokens up to 512, and
  num_heads up to 4 as the validated smoke envelope.
- The repo context notes that the exact no-patch seq128 score is already recorded and should not
  be repeated without a new candidate patch.

Verification:

- `uv run --extra cuda python -m avo compile --source candidates/cuda_mma_attention/attention_kernel.cu
  --out-dir build/manual_mma_seq128_head_dim128`: passed on sm86 with no spills, 40 registers,
  1 barrier, 18112 bytes shared memory, 400 bytes `cmem[0]`, 224 bytes `cmem[4]`, and
  28 bytes global memory.
- `uv run --extra cuda python -m avo score --backend candidate --candidate
  candidates/cuda_mma_attention_seed.py --seq-lens 128 --total-tokens 512 --num-heads 4
  --head-dim 128 --dtype bf16 --causal both --repeats 1 --warmup 1 --trials 3 --timeout-s 300`:
  all_correct true, geomean `0.1149927481033293` TFLOPS.
- `uv run --extra dev pytest tests/test_agent.py -q`: 76 passed.
- `uv run --extra dev pytest`: 165 passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

Score details:

- Noncausal: correct, max_abs_error `0.00390625`, median `0.8828799724578857 ms`,
  `0.1520226216326391` TFLOPS, samples
  `[0.8828799724578857, 0.7262719869613647, 0.9615039825439453]`.
- Causal: correct, max_abs_error `0.0078125`, median `0.7715200185775757 ms`,
  `0.08698266070104863` TFLOPS, samples
  `[0.7715200185775757, 0.8264960050582886, 0.6658560037612915]`.

Decision:

- This is structural correctness progress, not a lineage improvement. The workload signature is
  seq128/head_dim128 and remains incomparable to the current seq256/head_dim128 warp-row best.

## 2026-05-08 - Checkpoint 4.02: MMA seq256/head_dim128 lineage acceptance

Success criteria for this checkpoint:

- Run one bounded loop after the seq128 MMA proof.
- Preserve a compile-clean seq256 extension if it scores correctly.
- Accept the MMA source into lineage if it beats the current seq256/head_dim128 best.
- Align planner caps and notes so future work optimizes the accepted seq256 MMA seed rather than
  repeating the no-patch score.

Source refresh:

- Exa refreshed Ampere FlashAttention context before this sequence: NVIDIA's CUTLASS CuTeDSL
  Ampere FlashAttention v2 example uses `cp.async`, Ampere tensor-core MMA, register pipelining,
  online softmax, and output rescaling; FlashAttention-2's sm8x head_dim128 heuristic uses smaller
  N-blocks on sm86 in some cases. The practical takeaway is still to keep extending only
  correctness-proven MMA smoke envelopes before adding async-copy overlap.

Loop result:

- The agent proposed extending the MMA seed from seq_len 128 to seq_len 256 while keeping
  head_dim 128 and the eight 16-wide QK/PV WMMA chunks.
- The generated patch changed `kMaxSeqLen` to 256 and added 256 to `SMOKE_SEQUENCES`.
- The generated runtime check accidentally dropped seq32 from the accepted sequence list, so the
  manual version restored an explicit 16/32/64/128/256 check.
- The bounded step compiled only and was cleaned up successfully.

Manual runtime change:

- Reapplied the seq256 cap extension with the corrected runtime check.
- The MMA seed now accepts seq_len 16/32/64/128/256, head_dim 128, total_tokens up to 1024, and
  num_heads up to 4 as the validated smoke envelope.
- The repo context and score validator now treat the seq256 score as an accepted lineage result,
  and require compile-first validation for patched MMA shape extensions beyond seq256/head_dim128.

Lineage result:

- The fresh seq256 score passed correctness and was accepted into lineage as
  `evolve: accept mma seq256 head_dim128`.
- Previous best geomean was `0.43185073056556733` TFLOPS on the same seq256/head_dim128 BF16
  suite.
- Accepted geomean is `0.4924015757468769` TFLOPS.

Verification:

- `uv run --extra cuda python -m avo compile --source candidates/cuda_mma_attention/attention_kernel.cu
  --out-dir build/manual_mma_seq256_head_dim128`: passed on sm86 with no spills, 40 registers,
  1 barrier, 18112 bytes shared memory, 400 bytes `cmem[0]`, 224 bytes `cmem[4]`, and
  28 bytes global memory.
- `uv run --extra cuda python -m avo score --backend candidate --candidate
  candidates/cuda_mma_attention_seed.py --seq-lens 256 --total-tokens 1024 --num-heads 4
  --head-dim 128 --dtype bf16 --causal both --repeats 1 --warmup 1 --trials 3 --timeout-s 300`:
  all_correct true, geomean `0.4924015757468769` TFLOPS.
- `uv run --extra dev pytest tests/test_agent.py -q`: passed.
- `uv run --extra dev pytest`: passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

Score details:

- Noncausal: correct, max_abs_error `0.001953125`, median `0.7603840231895447 ms`,
  `0.7060523309629976` TFLOPS, samples
  `[0.7473919987678528, 0.8475520014762878, 0.7603840231895447]`, cv
  `0.056643140035337`.
- Causal: correct, max_abs_error `0.0078125`, median `0.7816960215568542 ms`,
  `0.3434013332514782` TFLOPS, samples
  `[0.7816960215568542, 0.6476799845695496, 0.8196799755096436]`, cv
  `0.09841028561612614`.

Decision:

- This is the first correctness-proven MMA smoke source that beat the previous seq256/head_dim128
  lineage best. It improves noncausal throughput materially and keeps causal throughput above the
  prior best case, but it is still a small smoke kernel rather than an FA2-level implementation.

## 2026-05-08 - Checkpoint 4.03: MMA K-staging compile-only probe

Success criteria for this checkpoint:

- Refresh Ampere attention-kernel references after the accepted seq256 MMA checkpoint.
- Run one bounded Anthropic loop from the current lineage.
- Record whether the next structural K/V staging step compiles, scores, or should be avoided as a
  repeated compile-only probe.
- Clean up any transient agent patch and keep local attempt records out of the committed runtime
  tree.

Source refresh:

- Exa found NVIDIA's CUTLASS CuTeDSL Ampere FlashAttention v2 example as the most relevant primary
  reference for the next direction: it uses `cp.async`, Ampere tensor-core MMA, register pipelining,
  shared-memory swizzles, online softmax, and output rescaling.
- Exa also found a LeetCUDA Ampere-style MMA attention kernel that uses a 64x64-ish warp-tiled
  structure, shared Q/K/V tiles, warp-level online softmax reductions, register reuse for P/V, and
  optional multi-stage K/V prefetching. It is useful as search-space evidence, not as code to copy.

Loop result:

- The agent proposed a conservative synchronous K-staging substrate for the current
  `cuda_mma_attention` seed.
- The patch added `__shared__ __nv_bfloat16 k_shared[kTile * kHeadDim]`, loaded the full 16x128 K
  tile cooperatively with all 256 threads, and used shared memory only for QK chunk 0.
- The other seven QK chunks and all eight PV chunks remained on the existing global-load path.
- The bounded command was compile-only:
  `avo compile --source candidates/cuda_mma_attention/attention_kernel.cu
  --out-dir build/mma_k_staging_chunk0`.

Verification:

- Compile passed on sm86 with no spills, 40 registers, 1 barrier, 22208 bytes shared memory,
  400 bytes `cmem[0]`, 224 bytes `cmem[4]`, 8 bytes `cmem[2]`, and 28 bytes global memory.
- The orchestrator cleaned up the transient candidate patch successfully.
- No score was run and no lineage gate decision was made.

Decision:

- Do not preserve the single-chunk staging patch as runtime code. It is only a compile proof for a
  shared-memory staging substrate and would add cooperative-load overhead while still leaving most
  QK/PV loads global.
- The agent's expected-effect text understated the shared-memory increase: a 16x128 BF16 K tile is
  4096 bytes, which matches the jump from 18112 to 22208 bytes.
- Runtime `.gitignore` now ignores `attempts/`, matching the design that attempt history is local
  cross-step memory rather than committed lineage.
- The next K/V staging step should either stage all QK chunks and score, or compile-check a clearly
  different `cp.async`/double-buffered path before scoring.

## 2026-05-08 - Checkpoint 4.04: Guard bad shared-K tile indexing

Success criteria for this checkpoint:

- Run one more bounded Anthropic step with the K-staging compile-only attempt in local memory.
- Determine whether the proposed full K-staging path is scoreable.
- If the decision is self-invalid, add a narrow validation guard and durable note instead of
  preserving or scoring the patch.

Loop result:

- The agent proposed synchronous shared-memory staging for the full 16x128 K tile.
- The patch cooperatively loaded `k_shared[kTile * kHeadDim]` and switched all eight QK K-fragment
  WMMA loads to shared memory.
- The bounded command was compile-only:
  `avo compile --source candidates/cuda_mma_attention/attention_kernel.cu
  --out-dir build/mma_k_staging_sync`.
- Compile passed on sm86 with no spills, 40 registers, 1 barrier, 22208 bytes shared memory,
  400 bytes `cmem[0]`, 224 bytes `cmem[4]`, 8 bytes `cmem[2]`, and 28 bytes global memory.
- Cleanup reverse-applied the transient patch successfully.

Decision:

- Do not score or preserve the patch. Its own risk text correctly identified the bug:
  `wmma::load_matrix_sync(k_frag, k_shared + key_start * kHeadDim + chunk_offset, kHeadDim)` uses a
  global key offset against a tile-local shared buffer.
- The correct base for the staged tile is `k_shared + chunk_offset` with leading dimension
  `kHeadDim`.
- Runtime validation now rejects candidate patches that stage an MMA K tile in shared memory and
  then load it with `k_shared + key_start * kHeadDim`.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.05: Synchronous K staging regression

Success criteria for this checkpoint:

- Run another bounded Anthropic step after the shared-K indexing guard.
- If it produces a corrected scoreable K-staging patch, compile and score it manually.
- Keep the patch only if it passes the lineage gate; otherwise clean it up and add a repeat guard.

Loop result:

- The agent proposed the corrected full synchronous K-staging patch:
  `k_shared + chunk_offset` with leading dimension `kHeadDim`, not the bad global
  `key_start * kHeadDim` offset.
- The bounded step compiled only and was cleaned up.
- Manual reapplication compiled cleanly with no spills, 40 registers, 1 barrier, 22208 bytes shared
  memory, 400 bytes `cmem[0]`, 224 bytes `cmem[4]`, 8 bytes `cmem[2]`, and 28 bytes global memory.

Score result:

- Command: `uv run --extra cuda python -m avo score --backend candidate --candidate
  candidates/cuda_mma_attention_seed.py --seq-lens 256 --total-tokens 1024 --num-heads 4
  --head-dim 128 --dtype bf16 --causal both --repeats 1 --warmup 1 --trials 3 --timeout-s 300`.
- Correctness passed for both causal modes.
- Geomean was `0.30611777431945414` TFLOPS, below the accepted best
  `0.4924015757468769` TFLOPS.
- Lineage gate rejected the candidate with reason `candidate regressed geomean throughput`.

Score details:

- Noncausal: max_abs_error `0.001953125`, median `1.2412480115890503 ms`,
  `0.4325250932830868` TFLOPS, samples
  `[1.1060160398483276, 1.2412480115890503, 1.3872640132904053]`.
- Causal: max_abs_error `0.0078125`, median `1.2390079498291016 ms`,
  `0.2166535380479405` TFLOPS, samples
  `[1.155616044998169, 1.2390079498291016, 1.2886719703674316]`.

Decision:

- Reverted the manual K-staging patch. Synchronous staging removes repeated K global loads but adds
  a cooperative copy and synchronization with no overlap, so it is slower.
- Runtime validation now rejects repeat static `k_shared` MMA K staging unless the patch adds real
  async-copy or double-buffered overlap.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.06: Synchronous V staging regression

Success criteria for this checkpoint:

- Run one bounded Anthropic step after the synchronous K-staging regression guard.
- If the proposed patch is scoreable, measure correctness and TFLOPS on the fixed
  seq256/head_dim128 BF16 suite before accepting or rejecting it.
- Keep the accepted kernel source unchanged if the patch regresses.
- Add a narrow planner guard and knowledge note only for the exact regressed pattern.

Source refresh:

- Exa found NVIDIA's CUTLASS Ampere FlashAttention v2 example again as the most relevant search
  reference: it combines `cp.async`, Ampere tensor-core MMA, register pipelining, online softmax,
  and output rescaling. Source:
  https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py
- NVIDIA's Ampere tuning guide frames async global-to-shared copy as the Ampere mechanism for
  explicit compute/data overlap while avoiding extra copy registers. Source:
  https://docs.nvidia.com/cuda/ampere-tuning-guide/
- NVIDIA's CUDA async-copy guide notes that async global-to-shared copies need a completion
  mechanism, and shared data consumed by other threads still needs synchronization after the copy
  completes. Source:
  https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html

Loop result:

- The agent proposed static double-buffered V staging for the current MMA seed:
  `v_shared[2][kTile * kHeadDim]`, cooperative warp loads of the current and next 16x128 BF16 V
  tiles, and PV `wmma::load_matrix_sync` from `v_shared[current_buffer]`.
- The bounded step compiled only and cleaned up the transient patch.
- Compile diagnostics were clean: no spills, 40 registers, 1 barrier, 26304 bytes shared memory,
  400 bytes `cmem[0]`, 224 bytes `cmem[4]`, and 28 bytes global memory.

Score result:

- Manual reapplication passed correctness for both causal modes.
- Command: `uv run --extra cuda python -m avo score --backend candidate --candidate
  candidates/cuda_mma_attention_seed.py --seq-lens 256 --total-tokens 1024 --num-heads 4
  --head-dim 128 --dtype bf16 --causal both --repeats 1 --warmup 1 --trials 3 --timeout-s 300`.
- Geomean was `0.31531656344385717` TFLOPS, below the accepted best
  `0.4924015757468769` TFLOPS.

Score details:

- Noncausal: max_abs_error `0.001953125`, median `1.320255994796753 ms`,
  `0.4066415256706702` TFLOPS, samples
  `[1.320255994796753, 1.4197759628295898, 1.2469120025634766]`.
- Causal: max_abs_error `0.0078125`, median `1.0978879928588867 ms`,
  `0.24450167753542637` TFLOPS, samples
  `[1.0978879928588867, 1.2736639976501465, 1.0158400535583496]`.

Decision:

- Reverted the manual V-staging patch. It preserved correctness but inserted synchronous shared
  memory traffic without useful overlap, so it was slower than direct global V loads.
- Runtime validation now rejects repeat static `v_shared[2]` MMA V staging unless the patch adds
  real async-copy overlap.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 79 tests.
- `uv run --extra dev pytest`: passed, 168 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.07: MMA probability-skew regression

Success criteria for this checkpoint:

- Run one bounded Anthropic step after the synchronous V-staging regression guard.
- Determine whether the proposed probability-buffer skew is compile-safe and scoreable.
- Preserve the accepted kernel only if the fixed candidate beats the current lineage best.
- Record invalid WMMA stride constraints and measured regression evidence.

Source refresh:

- Exa again surfaced NVIDIA's CUTLASS Ampere FlashAttention v2 example as the relevant target
  pattern: `cp.async`, tensor-core MMA, register pipelining, and online softmax rather than static
  synchronous shared-memory copies. Source:
  https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py
- NVIDIA's CUDA Programming Guide documents `wmma::load_matrix_sync` alignment requirements:
  the source pointer must be aligned and the leading dimension must be a 16-byte multiple for
  half-type multiplicands. Source:
  https://docs.nvidia.com/cuda/archive/13.0.3/cuda-c-programming-guide/index.html

Loop result:

- The agent proposed padding the MMA `probabilities` shared-memory tile from 16x16 to 16x17 and
  loading it with `wmma::load_matrix_sync(..., kTile + 1)`.
- The generated patch accidentally removed `kOutputElements`, so the compile command failed with
  `identifier "kOutputElements" is undefined`.
- The generated patch was also structurally invalid for WMMA: `kTile + 1` is 17 BF16 elements,
  not a 16-byte-aligned leading dimension.

Manual correction:

- Tested the same idea with an aligned stride: `kProbabilityStride = kTile + 8`, so each row is
  24 BF16 elements / 48 bytes.
- Updated both probability write sites, including the causal `valid_keys == 0` zero path, and used
  `kProbabilityStride` as the WMMA probability load leading dimension.
- Compile passed on sm86 with no spills, 40 registers, 1 barrier, 18368 bytes shared memory,
  400 bytes `cmem[0]`, 224 bytes `cmem[4]`, and 28 bytes global memory.

Score result:

- Correctness passed for both causal modes.
- Geomean was `0.46984155560491525` TFLOPS, below the accepted best
  `0.4924015757468769` TFLOPS.

Score details:

- Noncausal: max_abs_error `0.001953125`, median `0.8241599798202515 ms`,
  `0.6514158963616397` TFLOPS, samples
  `[0.993120014667511, 0.8241599798202515, 0.7926080226898193]`.
- Causal: max_abs_error `0.0078125`, median `0.7921280264854431 ms`,
  `0.3388788769297927` TFLOPS, samples
  `[0.6605759859085083, 0.7921280264854431, 1.000831961631775]`.

Decision:

- Reverted the manual probability-skew patch because it preserved correctness but regressed the
  fixed benchmark.
- Runtime validation now rejects `kTile + 1` probability WMMA leading dimensions and the measured
  stride-24 probability-buffer skew repeat.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 81 tests.
- `uv run --extra dev pytest`: passed, 170 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.08: Scalar async-copy planner rejection

Loop result:

- Ran one bounded `evolve-loop` step after the probability-skew regression guard.
- No candidate patch was applied and no compile or score command ran.
- The agent planner failed validation after three attempts because the proposed patch used scalar
  BF16 `__pipeline_memcpy_async` copies.

Decision:

- This is the expected guard behavior. Ampere async-copy candidates must use aligned 16-byte groups
  or a carefully justified tail path, not scalar 2-byte BF16 async copies.
- No runtime code change is needed; the existing validator already blocks this failure mode before
  patch application.

Verification:

- No runtime files changed.
- `git diff --check`: passed in the paper repo.

## 2026-05-08 - Checkpoint 4.09: Scalar async-copy retry feedback

Success criteria for this checkpoint:

- Inspect why the previous planner rejection repeated the same invalid scalar async-copy pattern.
- Keep the fix narrow: improve retry feedback, not the CUDA candidate search space.
- Verify the retry hint with a unit test and run the full runtime test suite.

Source refresh:

- Anthropic's strict tool-use docs reinforce keeping the decision boundary schema-constrained for
  production agent workflows. Source:
  https://console.anthropic.com/docs/en/agents-and-tools/tool-use/strict-tool-use
- Anthropic's tool-call handling docs recommend instructive error messages that explain what went
  wrong and what the model should try next, instead of generic failures. Source:
  https://console.anthropic.com/docs/en/agents-and-tools/tool-use/handle-tool-calls
- Anthropic's tool troubleshooting docs recommend more detailed tool descriptions and examples
  when tool use remains ambiguous. Source:
  https://console.anthropic.com/docs/en/agents-and-tools/tool-use/troubleshooting-tool-use

Implementation:

- Added a validation-feedback hint for scalar BF16 async-copy proposals.
- The retry message now explicitly says not to retry `sizeof(__nv_bfloat16)` async copies, explains
  that 16 bytes is 8 BF16 elements, and tells the planner to choose a materially different
  non-async patch if it cannot express a clean aligned-group async-copy diff.
- Added a focused unit test for that retry-feedback text.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 82 tests.
- `uv run --extra dev pytest`: passed, 171 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.10: No-op async helper wrapper rejection

Success criteria for this checkpoint:

- Run one bounded loop after the scalar async-copy retry feedback.
- Confirm whether the improved feedback moves the planner away from scalar BF16 async copies.
- Reject and record no-op async-copy API proofs that do not change real dataflow.
- Add narrow validation guards for the exact wrapper-only and malformed-signature failure modes.

Loop result:

- The planner no longer retried `sizeof(__nv_bfloat16)` scalar async copies.
- It instead proposed adding `#include <cuda_pipeline_primitives.h>` and three wrappers:
  `async_copy_16`, `async_commit`, and `async_wait`.
- The decision text stated the helpers were unused and that kernel logic was unchanged, so the
  patch could not affect correctness or throughput.
- The generated patch also inserted the helper definitions inside the
  `mma_attention_kernel` signature and duplicated the kernel declaration.

Compile result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_async_feedback.json`.
- The patch applied, then `avo compile --source candidates/cuda_mma_attention/attention_kernel.cu
  --out-dir build/mma_async_header_check` failed.
- NVCC reported an invalid parameter specifier at the inserted helper, an expected `)`, and
  undefined `dst_smem` / `src_global` identifiers.
- Cleanup reverse-applied the transient patch successfully.
- No score payload was produced and the lineage did not change.

Implementation:

- Runtime validation now rejects non-empty patches whose own text says an async-copy helper is
  unused.
- Runtime validation now rejects the wrapper-only `async_copy_16` / `async_commit` / `async_wait`
  API proof pattern when the wrappers are added but not called.
- Runtime validation now rejects async helper definitions inserted together with a duplicate
  `mma_attention_kernel` declaration.
- Runtime knowledge now records that wrapper-only async-copy API proofs are exhausted; future
  wrapper patches must be placed before the kernel signature and used in real aligned 16-byte-group
  dataflow.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 85 tests.
- `uv run --extra dev pytest`: passed, 174 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.11: Unpatched MMA rescore rejection

Success criteria for this checkpoint:

- Run one bounded loop after the no-op async helper wrapper guard.
- Confirm the planner avoids the rejected async-helper pattern.
- Reject any new loop waste discovered by that step.

Loop result:

- The planner avoided scalar async copies and wrapper-only async API proof patches.
- It chose no-edit mode and rescored the current unmodified MMA seed on the accepted
  seq256/head_dim128 BF16 suite to establish a "fresh baseline".
- The score passed correctness but regressed the accepted lineage best, so the gate rejected it.
- No patch was applied, no cleanup was needed, and the lineage did not change.

Score result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_async_wrapper_guard.json`.
- Candidate geomean: `0.3399665809331029` TFLOPS.
- Accepted best geomean: `0.4924015757468769` TFLOPS.
- Noncausal: max_abs_error `0.001953125`, median `1.0803519487380981 ms`,
  `0.4969407540080716` TFLOPS, samples
  `[1.0050560235977173, 1.0803519487380981, 1.1936639547348022]`.
- Causal: max_abs_error `0.0078125`, median `1.1541759967803955 ms`,
  `0.23257757633914394` TFLOPS, samples
  `[1.1541759967803955, 1.228991985321045, 1.0854719877243042]`.

Decision:

- The no-edit rescore confirmed a prompt-only note was not enough: the runtime validator still
  allowed recorded unpatched MMA seed scores under the current cap.
- Runtime validation now rejects unpatched MMA seed score commands and requires a structural
  `candidate_patch` before scoring that source again.
- Runtime knowledge records the rejected rescore and the updated guard.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 85 tests.
- `uv run --extra dev pytest`: passed, 174 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.12: Scalar async-copy retry hardening

Success criteria for this checkpoint:

- Refresh Ampere attention-copy guidance from primary sources before another agent step.
- Run one bounded loop after the unpatched MMA rescore guard.
- If the planner repeats scalar async-copy proposals, strengthen feedback without broadening the
  command or patch surface.

Source refresh:

- Exa found NVIDIA's CUTLASS Ampere FlashAttention v2 example again. The relevant pattern is
  still 128-bit global-to-shared copy atoms, swizzled Q/K/V shared layouts, Ampere tensor-core MMA,
  register staging, and integrated online-softmax rescaling. Source:
  https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py
- Exa also surfaced FlashAttention SM80 kernel traits showing `cute::uint128_t` copy atoms,
  BF16/FP16 SM80 tensor-core MMA, 8 BF16 values per global-copy vector, and smem layout choices for
  head_dim128. Source:
  https://github.com/vllm-project/flash-attention/blob/8798f277/csrc/flash_attn/src/kernel_traits.h

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_unpatched_mma_guard.json`.
- The planner failed validation after three attempts before any candidate command ran.
- The final validation error was the repeated scalar BF16 async-copy pattern:
  `candidate_patch uses scalar BF16 __pipeline_memcpy_async copies; use 16-byte aligned groups for
  Ampere async copy patches`.
- No patch was applied, no score payload was produced, and the lineage did not change.

Implementation:

- Strengthened retry feedback for scalar BF16 async-copy validation errors.
- The feedback now says the corrected decision should avoid `__pipeline_memcpy_async` entirely
  unless the diff contains real 16-byte-group dataflow, not wrapper/API proof code.
- Added the same scalar-async constraint to the repo context presented to the planner.
- Runtime knowledge records the primary-source takeaway: useful Ampere async-copy patches are
  vector-group dataflow changes coupled to MMA/online-softmax scheduling, not scalar BF16 copies.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 85 tests.
- `uv run --extra dev pytest`: passed, 174 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.13: Sync K-staging retry feedback

Success criteria for this checkpoint:

- Run one bounded loop after scalar async-copy retry hardening.
- Check whether the planner moves to a structurally different candidate.
- If it repeats a measured regression class during planning, improve targeted retry feedback.

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_scalar_retry_hardening.json`.
- The planner no longer failed on scalar BF16 async-copy validation.
- It instead failed validation after three attempts by repeating synchronous MMA K shared-memory
  staging.
- No candidate command ran, no patch was applied, no score payload was produced, and the lineage did
  not change.

Decision:

- The scalar async-copy hardening narrowed one repeated failure mode but exposed another: the planner
  still needed explicit retry feedback for the already measured sync-K regression.
- Runtime validation already rejected the bad patch, so the change is limited to clearer feedback.
- Added retry feedback saying not to retry static `k_shared` MMA K staging; a corrected K-staging
  patch must add real async-copy/double-buffered overlap, otherwise choose a different non-K-staging
  patch.
- Added matching feedback for the analogous static `v_shared[2]` MMA V-staging regression.
- Runtime knowledge records that later loops should not keep retrying static K staging.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 87 tests.
- `uv run --extra dev pytest`: passed, 176 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.14: Synchronous Q staging regression

Success criteria for this checkpoint:

- Run one bounded loop after sync-staging retry feedback.
- If the planner produces a materially different scoreable idea, test it manually if the generated
  patch only fails on stale context.
- Preserve the kernel only if it beats the accepted seq256/head_dim128 MMA best.

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_sync_retry_feedback.json`.
- The planner proposed synchronous shared-memory staging for the full 16x128 Q tile:
  `q_shared[kTile * kHeadDim]`, cooperative load once before the K loop, and QK
  `wmma::load_matrix_sync` from `q_shared + chunk_offset`.
- The generated patch failed `git apply --check` because its context was stale.
- No generated patch was applied by the orchestrator, and the lineage did not change.

Manual correction:

- Reapplied the Q-staging idea with current file context.
- Compile passed on sm86 with no spills, 40 registers, 1 barrier, 22208 bytes shared memory,
  400 bytes `cmem[0]`, 224 bytes `cmem[4]`, 8 bytes `cmem[2]`, and 28 bytes global memory.

Score result:

- Command: `uv run --extra cuda python -m avo score --backend candidate --candidate
  candidates/cuda_mma_attention_seed.py --seq-lens 256 --total-tokens 1024 --num-heads 4
  --head-dim 128 --dtype bf16 --causal both --repeats 1 --warmup 1 --trials 3 --timeout-s 300`.
- Correctness passed for both causal modes.
- Geomean was `0.42035719120740594` TFLOPS, below the accepted best
  `0.4924015757468769` TFLOPS.

Score details:

- Noncausal: max_abs_error `0.001953125`, median `0.9053760170936584 ms`,
  `0.5929811502224299` TFLOPS, samples
  `[0.8616639971733093, 0.9053760170936584, 1.761855959892273]`.
- Causal: max_abs_error `0.0078125`, median `0.9008319973945618 ms`,
  `0.2979861470023096` TFLOPS, samples
  `[0.9008319973945618, 0.8268160223960876, 0.9471359848976135]`.

Decision:

- Reverted the manual Q-staging patch. Like the prior synchronous K/V staging attempts, the static
  shared-memory copy and synchronization did not pay for itself without overlap.
- Runtime validation now rejects repeat static `q_shared` MMA Q staging unless the patch adds real
  overlap or a materially different Q/K dataflow.
- Runtime knowledge records the measured regression.

Verification:

- `git diff -- candidates/cuda_mma_attention/attention_kernel.cu`: empty after reverting the manual
  Q-staging patch.
- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 88 tests.
- `uv run --extra dev pytest`: passed, 177 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.15: Scalar async-copy cooldown prompt

Success criteria for this checkpoint:

- Run one bounded loop after rejecting the synchronous Q-staging regression.
- Determine whether blocking static Q/K/V staging moves the planner to a useful different patch.
- If scalar async-copy remains a repeated planning failure, make the base prompt constraint clearer.

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_sync_q_guard.json`.
- The planner again failed validation after three attempts on scalar BF16
  `__pipeline_memcpy_async` copies.
- No candidate command ran, no patch was applied, no score payload was produced, and the lineage did
  not change.

Decision:

- The scalar async-copy issue is no longer just a single retry-feedback problem; it keeps reappearing
  after unrelated guards.
- Added a base repo-context constraint that treats `cp.async` / `__pipeline_memcpy_async` as a
  cooled-down direction unless the diff is a complete aligned 16-byte-group dataflow change with
  exact current context and no scalar async calls.
- Runtime knowledge records the cooldown so future loops should favor non-async structural patches
  until the agent can express a valid vector-group async dataflow.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 88 tests.
- `uv run --extra dev pytest`: passed, 177 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.16: Thread-local row-state correctness guard

Success criteria for this checkpoint:

- Run one bounded loop after the scalar async-copy cooldown prompt.
- If the planner produces a non-async structural patch, let the existing score/correctness gate
  decide it.
- Record and guard any correctness-breaking pattern that makes it through patch validation.

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_scalar_async_cooldown.json`.
- The planner produced a non-async patch that moved MMA row softmax state from shared memory into
  per-thread register arrays.
- The patch applied and the candidate score command ran.
- Cleanup reverse-applied the patch successfully after rejection.
- The lineage did not change.

Score result:

- Both noncausal and causal cases failed correctness.
- Error for both cases: `RuntimeError: candidate output contains non-finite values`.
- Candidate geomean was `0.0` TFLOPS.
- Gate decision: rejected because the candidate failed correctness.

Decision:

- The patch's row-state ownership was wrong. It computed `reg_idx` from the row owner in one loop,
  then later used `row / blockDim.x` from unrelated output-scaling and final-store threads, so those
  threads read uninitialized per-thread row state.
- Runtime validation now rejects patches that move MMA row state into per-thread registers and then
  consume that state from cross-thread row loops.
- Runtime knowledge records the correctness failure and ownership constraint for any future
  register-row-state attempt.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 89 tests.
- `uv run --extra dev pytest`: passed, 178 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.17: Unused WMMA preload skeleton guard

Success criteria for this checkpoint:

- Run one bounded loop after the thread-local row-state guard.
- Preserve only patches that compile and have a plausible path to correctness/throughput impact.
- Clean up generated artifacts from compile-only attempts.

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_row_state_guard.json`.
- The planner proposed a QK software-pipeline skeleton with a `k_frag_next` WMMA matrix fragment.
- The patch loaded the next key tile's first K fragment into `k_frag_next`, but did not consume it
  in a later `wmma::mma_sync`.
- The compile command succeeded and cleanup reverted the patch. No score payload was produced, and
  the lineage did not change.

Compile result:

- Ptxas stayed at the accepted baseline resource footprint: no spills, 40 registers, 1 barrier,
  18112 bytes shared memory, 400 bytes `cmem[0]`, 224 bytes `cmem[4]`, and 28 bytes global memory.
- Because the command used `--out-dir candidates/cuda_mma_attention`, it left
  `candidates/cuda_mma_attention/attention_kernel.o`; removed that generated object file.

Decision:

- Treat the patch as a compile-only no-op. The unused preload cannot affect correctness or
  throughput and likely explains the unchanged resource counts.
- Runtime validation now rejects patches that add and load an MMA preload fragment without consuming
  it in MMA dataflow.
- Runtime validation now requires `avo compile --out-dir` to be under `build/`, preventing generated
  object files from being written into `candidates/`.
- Runtime knowledge records both rules.

Verification:

- `rm -f candidates/cuda_mma_attention/attention_kernel.o`: removed the generated object file.
- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 91 tests.
- `uv run --extra dev pytest`: passed, 180 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.18: Single-buffer V staging repeat guard

Success criteria for this checkpoint:

- Run one bounded loop after the unused WMMA preload skeleton guard.
- If the planner repeats an already measured regression with minor spelling changes, generalize the
  guard rather than preserving or scoring it.

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_preload_guard.json`.
- The planner proposed static synchronous V staging again, but as a single-buffer
  `v_shared[kTile * kHeadDim]` compile-only variant.
- The patch applied, compiled, and cleanup reverted it.
- No score payload was produced and the lineage did not change.

Compile result:

- Compile passed on sm86 with no spills, 40 registers, 1 barrier, 22208 bytes shared memory,
  400 bytes `cmem[0]`, 224 bytes `cmem[4]`, and 28 bytes global memory.

Decision:

- This is the same static synchronous V-staging direction already measured as a throughput
  regression, only with a single buffer and compile-only command.
- Runtime validation now rejects both `v_shared[2][kTile * kHeadDim]` and
  `v_shared[kTile * kHeadDim]` repeats when consumed synchronously by PV WMMA without async overlap.
- Runtime knowledge records the single-buffer repeat.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 92 tests.
- `uv run --extra dev pytest`: passed, 181 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.19: Self-invalid patch retry feedback

Success criteria for this checkpoint:

- Run one bounded loop after the single-buffer V-staging repeat guard.
- If planning fails before command execution, keep the fix at the decision boundary.
- Preserve the existing candidate source and lineage unless a patch clears validation and scoring.

Source refresh:

- Exa refreshed Ampere/SM80 attention references before this run. The relevant reinforcement was
  unchanged: production Ampere paths use register fragments, online-softmax rescaling, and tuned
  block sizes; they do not justify no-op skeletons or self-invalid patches.

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_single_v_guard.json`.
- The planner failed validation after three attempts.
- Final validation error: `candidate_patch is described as known invalid by the decision itself;
  found phrase 'will cause a compile error'`.
- No command ran, no patch was applied, no score payload was produced, and the lineage did not
  change.

Decision:

- The self-invalid patch validator is doing the right thing, but the retry feedback was too generic.
- Added targeted retry feedback: do not retry patches whose own hypothesis, expected effect, or risk
  says they will fail compile, break correctness, are unused, or are not ready. Return a corrected
  raw diff with the flaw removed, or switch to `No edit;` diagnostic mode.
- Runtime knowledge records the self-invalid planning failure.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 93 tests.
- `uv run --extra dev pytest`: passed, 182 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.20: Compile out-dir retry feedback

Success criteria for this checkpoint:

- Run one bounded loop after self-invalid patch retry feedback.
- If the planner fails only on command-shape validation, keep the fix in retry feedback rather than
  changing CUDA sources.
- Keep generated compiler artifacts out of `candidates/`.

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_self_invalid_feedback.json`.
- The planner failed validation after three attempts.
- Final validation error: `next_command --out-dir must be under: build`.
- No command ran, no patch was applied, no score payload was produced, and the lineage did not
  change.

Decision:

- The new `--out-dir` guard correctly prevents compile artifacts from being written into
  `candidates/`, but retry feedback needed to tell the agent how to fix the command.
- Added targeted retry feedback telling compile checks to use a repo-relative `build/<name>`
  directory and not write compiler outputs under `candidates/`.
- Runtime knowledge records the repeated invalid out-dir failure.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 94 tests.
- `uv run --extra dev pytest`: passed, 183 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.21: Malformed Q-preload fragment guard

Success criteria for this checkpoint:

- Run one bounded loop after compile out-dir retry feedback.
- Let compile/check gates handle any patch that clears decision validation.
- If compile fails due to a syntactic pattern the decision already warned about, tighten the
  decision validator for that concrete pattern.

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_out_dir_feedback.json`.
- The planner proposed a Q-fragment preload patch and used a valid `--out-dir build/mma_q_preload`.
- The patch applied, then compile failed.
- Cleanup reverse-applied the patch successfully.
- No score payload was produced and the lineage did not change.

Compile result:

- NVCC rejected `wmma::fragment<wmma::matrix_a, kTile, kTile, 16, wmma::row_major> q_frag_chunk`
  because the matrix fragment omitted the required element type.
- The decision risk text also said `Do not use this diff`, so it should have been rejected as
  self-invalid before patch application.

Decision:

- Added `do not use this diff` and `would fail compilation` to the self-rejecting phrase list.
- Runtime validation now rejects WMMA matrix A/B fragment declarations that jump directly from the
  K dimension to `wmma::row_major` / `wmma::col_major` without an element type such as
  `__nv_bfloat16`.
- Runtime knowledge records the malformed Q-preload fragment failure.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 96 tests.
- `uv run --extra dev pytest`: passed, 185 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.22: Regressed QK preload chain guard

Success criteria for this checkpoint:

- Run one bounded loop after the malformed Q-preload fragment guard.
- If the agent finds a compile-clean structural QK scheduling patch, score it against the current
  seq256/head_dim128 BF16 lineage signature before deciding whether it belongs in lineage.
- Preserve the current lineage unless the score gate improves on the accepted best.

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_q_preload_guard.json`.
- The planner proposed a QK WMMA fragment software-pipeline patch using a `k_frag_next` matrix-B
  fragment. It loaded chunk 0 before the QK chunk loop, consumed `k_frag_next` in `mma_sync`, and
  loaded `next_chunk` at the end of each chunk.
- The bounded loop used a compile command, so it produced no score payload. Cleanup reverse-applied
  the patch successfully and the lineage did not change.

Compile result:

- The loop compile passed on sm86 with no spills, 40 registers, 1 barrier, 18112 bytes shared
  memory, 400 bytes `cmem[0]`, 224 bytes `cmem[4]`, and 28 bytes global memory.
- A manual reapply of the same patch compiled with the same resource shape.

Manual score result:

- Command: `uv run --extra cuda python -m avo score --backend candidate --candidate
  candidates/cuda_mma_attention_seed.py --seq-lens 256 --total-tokens 1024 --num-heads 4
  --head-dim 128 --dtype bf16 --causal both --repeats 1 --warmup 1 --trials 3 --timeout-s 300`.
- Correctness passed for both causal modes.
- Geomean regressed to `0.4462305013884498` TFLOPS versus the current best
  `0.4924015757468769`.
- Noncausal: max error `0.001953125`, median `0.9140160083770752` ms,
  `0.5873758304882064` TFLOPS.
- Causal: max error `0.0078125`, median `0.7918400168418884` ms,
  `0.33900213463649714` TFLOPS.

Decision:

- Reverted the temporary candidate source edit after scoring.
- Runtime validation now rejects the exact QK `k_frag_next` preload chain that compiled and
  preserved correctness but reduced geomean throughput.
- Runtime knowledge records the score regression so future QK scheduling work must be materially
  different or profiler-driven.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 97 tests.
- `uv run --extra dev pytest`: passed, 186 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.23: Unpatched MMA score retry feedback

Success criteria for this checkpoint:

- Run one bounded loop after the regressed QK preload chain guard.
- If planning fails before command execution, keep the fix at the decision retry boundary.
- Preserve the candidate source and lineage unless a patch clears validation and scoring.

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_qk_preload_chain_guard.json`.
- The planner failed validation after three attempts.
- Final validation error: `next_command repeats a recorded unpatched MMA seed score; include
  candidate_patch to change kernel structure before scoring`.
- No command ran, no patch was applied, no score payload was produced, and the lineage did not
  change.

Decision:

- The unpatched-score validator is correct: the accepted MMA seq256/head_dim128 score is already
  in lineage and should not be repeated as a no-edit candidate step.
- Added targeted retry feedback telling the planner not to retry no-edit
  `cuda_mma_attention_seed.py` scores. To score that candidate again, it must provide a structural
  raw diff under `candidates/cuda_mma_attention/` or choose a different diagnostic.
- Runtime knowledge records the repeated invalid unpatched-score failure.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 98 tests.
- `uv run --extra dev pytest`: passed, 187 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.24: Self-invalid PV preload retry guard

Success criteria for this checkpoint:

- Run one bounded loop after unpatched MMA score retry feedback.
- If a patch applies but its own risk text predicts a compile failure, move the fix to decision
  validation so the patch is not applied in future loops.
- Keep the current candidate source and lineage unchanged unless a patch clears validation and
  scoring.

Source refresh:

- Exa refreshed Ampere/SM80 attention references before this run. The useful reinforcement was that
  real Ampere FlashAttention-style kernels combine `cp.async`, tensor-core MMA, online softmax, and
  register/shared-memory pipelines. The search did not justify repeating self-invalid preload
  patches or extra shared-memory round trips without materially different dataflow.

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_unpatched_score_feedback.json`.
- The planner proposed a PV `v_frag_next` preload patch and used
  `--out-dir build/mma_pv_preload`.
- The patch applied, then compile failed.
- Cleanup reverse-applied the patch successfully.
- No score payload was produced and the lineage did not change.

Compile result:

- NVCC rejected the patch with two scope errors:
  `identifier "chunk_offset" is undefined` at the `pv_tile` store, and
  `identifier "chunk" is undefined` in a preload guard after the loop.
- The decision risk text explicitly warned that the patch referenced `chunk` outside the loop, would
  fail compilation, and should not be scored as-is.

Decision:

- Added `will fail compile`, `will fail compilation`, and `do not score this patch` to the
  self-rejecting phrase list.
- Runtime validation now rejects patches whose own decision text says they will fail compilation or
  should not be scored.
- Runtime knowledge records the PV-preload scope failure.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 99 tests.
- `uv run --extra dev pytest`: passed, 188 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.25: Accepted PV direct accumulation

Success criteria for this checkpoint:

- Run one bounded loop after the self-invalid PV preload guard.
- If the agent suggests a structurally useful but stale patch, manually test only the smallest
  corrected version against the current seq256/head_dim128 BF16 lineage signature.
- Accept into lineage only if correctness passes and geomean improves over the current best.

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_pv_self_invalid_guard.json`.
- The planner proposed a PV direct-accumulation patch, but its patch context was stale and
  `git apply --check` rejected it at `candidates/cuda_mma_attention/attention_kernel.cu:106`.
- No command ran, no score payload was produced, and the candidate source/lineage initially stayed
  unchanged.

Manual correction:

- Applied the intended minimal dataflow change directly:
  - Removed the intermediate float `pv_tile[kOutputElements]` shared-memory buffer.
  - Kept the existing `output_acc *= old_scale[row]` rescale before the PV loop.
  - Loaded each 16-column `output_acc` chunk into the WMMA accumulator fragment.
  - Ran PV `mma_sync` into that accumulator and stored it directly back to `output_acc`.
- This preserves the mathematical operation while removing the later `output_acc += pv_tile` loop.

Compile result:

- Command: `uv run --extra cuda python -m avo compile --source
  candidates/cuda_mma_attention/attention_kernel.cu --out-dir build/mma_pv_direct_accum_manual`.
- Compile passed on sm86 with no spills, 40 registers, 1 barrier, 9920 bytes shared memory,
  400 bytes `cmem[0]`, 224 bytes `cmem[4]`, and 28 bytes global memory.

Score result:

- Command: `uv run --extra cuda python -m avo score --backend candidate --candidate
  candidates/cuda_mma_attention_seed.py --seq-lens 256 --total-tokens 1024 --num-heads 4
  --head-dim 128 --dtype bf16 --causal both --repeats 1 --warmup 1 --trials 3 --timeout-s 300`.
- First score passed correctness and improved geomean to `0.49959324419420076` TFLOPS.
- Persisted rerun passed correctness and improved geomean to `0.5772885607891738` TFLOPS.
- Persisted noncausal: max error `0.001953125`, median `0.6164159774780273` ms,
  `0.8709555423863704` TFLOPS.
- Persisted causal: max error `0.0078125`, median `0.7015359997749329` ms,
  `0.382639602366977` TFLOPS.

Lineage decision:

- Accepted into lineage as commit `845ab85` (`evolve: accept mma pv direct accumulation`).
- Gate payload: candidate geomean `0.5772885607891738`, previous best
  `0.4924015757468769`, reason `candidate passed correctness and throughput gate`.
- Runtime source now matches the accepted direct-accumulation candidate.

Verification:

- `uv run --extra dev pytest`: passed, 188 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.26: Attempt-history payload filtering

Success criteria for this checkpoint:

- Prevent loop summary JSON files and manual score captures in `attempts/` from polluting the agent's
  recent-attempt prompt.
- Preserve the existing per-step attempt summaries and supervisor signals for real `EvolutionStep`
  records.
- Verify the fix with a regression test and the full runtime suite.

Observation:

- The `attempts/` directory intentionally contains both timestamped per-step records and local
  convenience files such as `loop_after_*.json`.
- `summarize_attempt_history` previously considered every `*.json` before applying the recent
  history limit, so loop summaries and manual score captures could crowd out real step records and
  produce placeholder text like `<missing command>` / `<missing hypothesis>`.
- This matched earlier planner confusion about recent attempts sharing missing command/hypothesis
  fields.

Decision:

- Runtime attempt-history summarization now filters for real step payloads before applying the
  recent-attempt limit.
- A valid step payload must have an `attempt` object with both `decision` and `command_result`
  objects.
- Loop summary JSON and score-wrapper JSON are ignored by the prompt summary and supervisor signal.

Verification:

- Added `test_summarize_attempt_history_ignores_loop_and_score_json`.
- `uv run --extra dev pytest tests/test_evolve.py -q`: passed, 37 tests.
- `uv run --extra dev pytest`: passed, 189 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in the runtime repo.
- Manual summary check on the live `attempts/` directory now reports only timestamped step records
  and no `<missing command>` placeholders.

## 2026-05-08 - Checkpoint 4.27: 2D probability-stride repeat guard

Success criteria for this checkpoint:

- Run one bounded loop after attempt-history filtering.
- If the planner repeats a previously rejected probability-buffer skew direction, reject that family
  more generally at decision validation.
- Preserve the accepted PV direct-accumulation source and lineage unless a candidate clears
  correctness and throughput gates.

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_attempt_history_filter.json`.
- The planner proposed probability-buffer stride padding with
  `kProbabilityStride = kTile + 8`, a 2D
  `probabilities[kTile][kProbabilityStride]` declaration, and a PV WMMA load from
  `&probabilities[0][0]`.
- The patch applied and the score command ran, but the score failed correctness.
- Cleanup reverse-applied the patch successfully and the lineage did not change.

Failure detail:

- The noncausal case failed during extension build:
  `expression must be a modifiable lvalue` at
  `probabilities[row * kTile + key] = __float2bfloat16(0.0f);`.
- The patch updated the regular probability store to 2D indexing, but left the masked-tile zero-fill
  branch using stale flattened indexing.
- The causal case produced a score in the same payload, but the overall score was rejected because
  `all_correct` was false.

Decision:

- This is still the same simple probability-buffer stride-24 skew direction already recorded as a
  throughput regression before the accepted PV direct-accumulation patch.
- Runtime validation now rejects both flat and 2D variants of the stride-24 probability-buffer skew
  repeat.
- Runtime knowledge records the post-direct-accum 2D repeat and stale flat-index compile failure.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 100 tests.
- `uv run --extra dev pytest`: passed, 190 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.28: Incomplete score-skew patch guard

Success criteria for this checkpoint:

- Run one bounded loop after the 2D probability-stride repeat guard.
- If a patch's own risk text says the diff is incomplete or should be rejected, catch that at
  decision validation.
- Reject CUDA patches that use 2D `probabilities[row][key]` indexing without declaring
  `probabilities` as a 2D shared tile.

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_probability_2d_guard.json`.
- The planner proposed a score-tile skew compile check using `scores[kTile][kScoreStride]`.
- The patch applied, then compile failed.
- Cleanup reverse-applied the patch successfully.
- No score payload was produced and the lineage did not change.

Compile result:

- NVCC rejected `probabilities[row][key] = __float2bfloat16(weight);` because `probabilities`
  remained a flat `__nv_bfloat16` shared array.
- The decision risk text also said the patch was incomplete and should be rejected before apply.

Decision:

- Added `diff is incomplete` and `should be rejected` to the self-rejecting phrase list.
- Runtime validation now rejects `probabilities[row][key]` patches unless the diff also declares
  `probabilities` as a 2D shared-memory tile.
- Runtime knowledge records the incomplete score-skew/probability-index failure.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 101 tests.
- `uv run --extra dev pytest`: passed, 191 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.29: Non-improving self-invalid retry feedback

Success criteria for this checkpoint:

- Run one bounded loop after the incomplete score-skew patch guard.
- If planning fails before command execution because the decision describes its own patch as unable
  to improve throughput, keep the fix in retry feedback.
- Preserve the candidate source and lineage.

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_score_skew_guard.json`.
- The planner failed validation after three attempts.
- Final validation error: `candidate_patch is described as known invalid by the decision itself;
  found phrase 'cannot improve throughput'`.
- No command ran, no patch was applied, no score payload was produced, and the lineage did not
  change.

Decision:

- The self-invalid validator was correct, but retry feedback did not explicitly name
  `cannot improve throughput` as a self-invalid patch description.
- Updated retry feedback so a patch whose own hypothesis, expected effect, or risk says it cannot
  improve throughput must be replaced with a corrected raw diff or a no-edit diagnostic.
- Runtime knowledge records the planning failure and retry-feedback adjustment.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 101 tests.
- `uv run --extra dev pytest`: passed, 191 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.30: Regressed score-tile skew guard

Success criteria for this checkpoint:

- Run one bounded loop after non-improving self-invalid retry feedback.
- Let a corrected score-tile skew patch go through score gating if it clears validation.
- If it preserves correctness but regresses throughput, record the score and reject exact repeats.

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_non_improving_feedback.json`.
- The planner proposed a corrected score-tile skew:
  `kScoreStride = 24`, `scores[kTile][kScoreStride]`, score WMMA store to
  `&scores[0][0]`, and 2D `scores[row][key]` reads.
- The patch applied, scored, and cleanup reverse-applied it successfully.
- The lineage did not change.

Score result:

- Correctness passed for both causal modes.
- Geomean regressed to `0.5374411946079206` TFLOPS versus the current best
  `0.5772885607891738`.
- Noncausal: max error `0.001953125`, median `0.7266560196876526` ms,
  `0.7388240067573786` TFLOPS.
- Causal: max error `0.0078125`, median `0.6866239905357361` ms,
  `0.39094971876900797` TFLOPS.
- Gate decision: rejected, reason `candidate regressed geomean throughput`.

Decision:

- Runtime validation now rejects exact MMA score-tile stride-24 skew repeats.
- Runtime knowledge records the correctness-preserving throughput regression.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 102 tests.
- `uv run --extra dev pytest`: passed, 192 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.31: Regressed probability stride-20 guard

Success criteria for this checkpoint:

- Run one bounded loop after the regressed score-tile skew guard.
- If a probability-buffer padding candidate passes correctness but regresses throughput, reject it
  through the lineage gate and record the exact pattern.
- Preserve the accepted direct-accumulation lineage unless the score improves.

Source refresh:

- Exa refreshed SM80/FlashAttention references before this run. The useful reinforcement was that
  Ampere paths rely on standard `cp.async`, warp-level MMA, online softmax, register-resident
  accumulators, and ordinary shared memory; the search did not justify repeating simple
  shared-memory padding without a better dataflow reason.

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_score_stride_guard.json`.
- The planner proposed a stride-20 probability tile:
  `kProbabilityStride = 20`, `probabilities[kTile][kProbabilityStride]`, 2D probability stores, and
  PV WMMA load from `&probabilities[0][0]`.
- The patch applied, scored, and cleanup reverse-applied it successfully.
- The lineage did not change.

Score result:

- Correctness passed for both causal modes.
- Geomean regressed to `0.5257029292160739` TFLOPS versus the current best
  `0.5772885607891738`.
- Noncausal: max error `0.001953125`, median `0.8048959970474243` ms,
  `0.6670065623004553` TFLOPS.
- Causal: max error `0.0078125`, median `0.6478719711303711` ms,
  `0.41433410914759705` TFLOPS.
- Gate decision: rejected, reason `candidate regressed geomean throughput`.

Decision:

- Runtime validation now rejects exact stride-20 probability-buffer skew repeats.
- Runtime knowledge records that simple probability padding remains unhelpful after PV direct
  accumulation.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 103 tests.
- `uv run --extra dev pytest`: passed, 193 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.32: Regressed Q-fragment preload guard

Success criteria for this checkpoint:

- Run one bounded loop after the stride-20 probability guard.
- Let a structurally new QK scheduling patch score if it clears validation.
- If it preserves correctness but regresses throughput, record the score and reject exact repeats.

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_probability_stride20_guard.json`.
- The planner proposed a Q-fragment software-pipeline patch using `q_frag_next`, assigning
  `q_frag = q_frag_next`, and loading the next Q fragment at the end of each QK chunk.
- The patch applied, scored, and cleanup reverse-applied it successfully.
- The lineage did not change.

Score result:

- Correctness passed for both causal modes.
- Geomean regressed to `0.4215966561800913` TFLOPS versus the current best
  `0.5772885607891738`.
- Noncausal: max error `0.001953125`, median `0.9138240218162537` ms,
  `0.5874992330940834` TFLOPS.
- Causal: max error `0.0078125`, median `0.8872640132904053` ms,
  `0.3025429319560828` TFLOPS.
- Gate decision: rejected, reason `candidate regressed geomean throughput`.

Decision:

- Runtime validation now rejects exact QK `q_frag_next` preload-chain repeats.
- Runtime knowledge records the correctness-preserving throughput regression.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 104 tests.
- `uv run --extra dev pytest`: passed, 194 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.33: Missing decision-key retry feedback

Success criteria for this checkpoint:

- Run one bounded loop after the Q-fragment preload guard.
- If planning fails before command execution because the agent returns a partial structured
  decision, keep the fix in retry feedback.
- Preserve the candidate source and lineage.

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_q_preload_guard_final.json`.
- The planner failed validation after three attempts.
- Final validation error: `variation decision missing required keys: expected_effect, risk,
  next_command`.
- No command ran, no patch was applied, no score payload was produced, and the lineage did not
  change.

Decision:

- Added targeted retry feedback telling the planner to return one complete variation decision object
  with every required field: `hypothesis`, `files_to_inspect`, `candidate_edit`, `candidate_patch`,
  `expected_effect`, `risk`, and `next_command`.
- The feedback explicitly says not to omit `expected_effect`, `risk`, or `next_command` even in
  no-edit diagnostic mode.
- Runtime knowledge records the partial-decision planning failure.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 105 tests.
- `uv run --extra dev pytest`: passed, 195 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.34: Compile-only WMMA skeleton guard

Success criteria for this checkpoint:

- Run one bounded loop after missing decision-key retry feedback.
- If the planner spends the step on a compile-only WMMA skeleton with no correctness or throughput
  evidence, preserve the candidate source and add a narrow guard against the no-op pattern.
- Keep the accepted direct-accumulation MMA lineage unchanged.

Source refresh:

- Exa refreshed the Dao-AILab SM80 FlashAttention mainloop before this run. The relevant
  reinforcement was that the Ampere path uses `cp.async` staging, online softmax, O in
  register-resident accumulator fragments, score rescaling, and PV accumulation directly into the
  O accumulator. Source:
  https://github.com/Dao-AILab/flash-attention/blob/main/hopper/mainloop_fwd_sm80.hpp

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_missing_key_feedback.json`.
- The planner patched the warp-row seed with `mma.h`, shared BF16 Q/K buffers, and a
  `wmma::accumulator` fragment, then ran a compile-only command.
- NVCC succeeded, but the patch never ran `mma_sync` and never connected the fragment to the
  online-softmax or PV dataflow. No correctness or throughput score was produced.
- Cleanup reverse-applied the patch successfully. The source tree and lineage remained unchanged.

Compile diagnostics:

- BF16/Half instantiations used 48 registers, 1 barrier, 17024 bytes shared memory, 412 bytes
  `cmem[0]`, and 224 bytes `cmem[4]`.
- The float instantiation used 56 registers, 1 barrier, and 33536 bytes shared memory.
- NVCC warned that `wmma_scores` was declared but never referenced.

Decision:

- Runtime validation now rejects patches whose own decision text says they do not affect correctness
  or throughput.
- Runtime validation now rejects WMMA compile skeletons that add fragments and `fill_fragment`
  without any `mma_sync` or online-softmax dataflow.
- Runtime knowledge records the compile-only WMMA detour and source refresh.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 107 tests.
- `uv run --extra dev pytest`: passed, 197 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.35: Malformed PV preload guard

Success criteria for this checkpoint:

- Run one bounded loop after the compile-only WMMA skeleton guard.
- If the planner proposes a structurally malformed PV preload patch and the decision text already
  identifies the compile failure, reject that class before future patch application.
- Preserve the accepted direct-accumulation source and lineage.

Source refresh:

- Exa refreshed Ampere FlashAttention references before this run. The useful reinforcement from
  NVIDIA CUTLASS and Dao-AILab sources was 128-thread SM80-style tiles, 16-byte `cp.async` copies,
  online softmax, O rescaling before PV, and register-resident O accumulation. Source:
  https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_wmma_skeleton_guard.json`.
- The planner proposed a PV probability-fragment preload using `probability_frag_next`.
- The patch applied, but compile failed and cleanup reverse-applied it successfully.
- No score payload was produced, no gate decision ran, and lineage stayed unchanged.

Compile failure:

- NVCC warned that a standalone `probability_frag;` statement had no effect.
- NVCC then failed because a stale `wmma::store_matrix_sync(&output_acc[chunk_offset], output_frag,
  ...)` line remained outside the PV chunk scope, leaving `chunk_offset` and `output_frag`
  undefined and causing follow-on parse errors.
- The decision risk text had already said the duplicate `probability_frag;` line and duplicate store
  line were a diff structure error that would cause NVCC compile failure.

Decision:

- Runtime validation now treats `will cause NVCC compile failure`, `diff structure error`, and
  `duplicate store line` as self-invalid patch descriptions.
- Runtime validation also rejects PV preload patches that add `probability_frag_next`,
  `probability_frag = probability_frag_next;`, and a standalone `probability_frag;` statement in
  added lines.
- Runtime knowledge records the malformed PV preload compile failure.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 109 tests.
- `uv run --extra dev pytest`: passed, 199 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.36: Unused Q-preload skeleton guard

Success criteria for this checkpoint:

- Run one bounded loop after the malformed PV preload guard.
- If the planner uses the step for a compile-only Q preload skeleton that is not consumed by the
  QK dataflow, record it and reject exact no-op skeletons before future compile steps.
- Preserve the accepted direct-accumulation lineage.

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_pv_preload_guard.json`.
- The planner proposed a Q double-buffer skeleton that declared `q_frag_next` and loaded the next
  query tile after the final key tile.
- NVCC compiled the patch successfully, cleanup reverse-applied it successfully, and no score
  payload was produced.
- The patch did not swap or consume `q_frag_next` in the QK MMA path. The decision text explicitly
  called it a compile-only structural probe that was not expected to improve throughput and must not
  be scored.

Compile diagnostics:

- The patched source matched the accepted kernel resource counts: no spills, 40 registers,
  1 barrier, 9920 bytes shared memory, 400 bytes `cmem[0]`, 224 bytes `cmem[4]`, and 28 bytes
  global memory.

Decision:

- Runtime validation now treats `compile-only structural probe`, `does not yet consume`, and
  `must not be scored` as self-invalid patch descriptions.
- Runtime validation now rejects unused `q_frag_next` and `probability_frag_next` preload skeletons
  when the patch adds a `load_matrix_sync(..._frag_next, ...)` but no MMA consumption.
- Runtime knowledge records the compile-only Q double-buffer skeleton.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 111 tests.
- `uv run --extra dev pytest`: passed, 201 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.37: Thread-0 row-state register guard

Success criteria for this checkpoint:

- Run one bounded loop after the unused Q-preload skeleton guard.
- If the planner proposes another unsafe register-row-state direction, reject the concrete pattern
  before future compile or score steps.
- Preserve the accepted direct-accumulation source and lineage.

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_q_preload_skeleton_guard.json`.
- The planner proposed moving `row_max`, `row_sum`, and `old_scale` from shared memory into local
  per-thread arrays.
- The patch did not apply: `git apply --check` failed with `error: corrupt patch at line 46`.
- No compile, score, cleanup, or lineage update was needed because the patch never applied.

Decision:

- The direction also repeated a known unsafe register-row-state idea in a new shape:
  `float row_max[kTile]`, `float row_sum[kTile]`, and `float old_scale[kTile]` were local per-thread
  arrays, but only `threadIdx.x == 0` initialized them. Other row-owning threads would read their
  own uninitialized local arrays if the patch compiled.
- Runtime validation now rejects local MMA row-state arrays paired with `threadIdx.x == 0`
  initialization as another per-thread-register row-state hazard.
- Runtime knowledge records the corrupt patch and the row-state ownership issue.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 112 tests.
- `uv run --extra dev pytest`: passed, 202 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.38: Recorded warp-row diagnostic guard

Success criteria for this checkpoint:

- Run one bounded loop after the thread-0 row-state guard.
- If the planner chooses a no-edit diagnostic score, preserve the result and reject exact repeats
  once it has served its purpose.
- Keep the accepted direct-accumulation MMA lineage unchanged.

Source refresh:

- Exa refreshed Ampere FlashAttention implementation notes before this run. The useful steering
  point was that current SM80-family FA-style paths use Ampere tensor cores, 16-byte `cp.async`,
  online softmax, and commonly choose larger 128x64-ish forward tiles for SM8x; this supports moving
  away from repeated 16x16 no-op preload skeletons and toward structural dataflow changes. Sources:
  https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py
  and https://github.com/Dao-AILab/flash-attention/blob/3387de49/hopper/mainloop_fwd_sm80.hpp

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_thread0_row_state_guard.json`.
- The planner chose no-edit mode and scored the existing warp-row seed on the fixed
  seq256/head_dim128 BF16 suite.
- Correctness passed for both causal modes, but the gate rejected the candidate because geomean
  regressed versus the accepted MMA direct-accumulation kernel.
- No patch was applied and lineage stayed unchanged.

Score result:

- Geomean: `0.4114026673771631` TFLOPS versus accepted MMA best `0.5772885607891738`.
- Noncausal: max error `0.001953125`, median `0.9919040203094482` ms,
  `0.5412528843592248` TFLOPS.
- Causal: max error `0.015625`, median `0.8584319949150085` ms,
  `0.31270439311453807` TFLOPS.
- Gate decision: rejected, reason `candidate regressed geomean throughput`.

Decision:

- Runtime validation now rejects exact no-patch warp-row seq256/head_dim128/1024-token/4-head
  scores unless a structural `candidate_patch` changes the kernel or wrapper first.
- Runtime retry feedback explains that the no-edit warp-row diagnostic is already recorded and
  gate-rejected.
- Runtime knowledge records the score and the instruction not to repeat it.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 114 tests.
- `uv run --extra dev pytest`: passed, 204 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.39: Must-not-score retry feedback

Success criteria for this checkpoint:

- Run one bounded loop after the recorded warp-row diagnostic guard.
- If planning fails because the agent repeatedly returns a self-invalid patch whose own text says it
  must not be scored, improve retry feedback rather than touching candidate kernels.
- Preserve source and lineage.

Source refresh:

- Exa refreshed SM8x FlashAttention launch choices before this run. The useful note was that
  upstream FA2 code comments identify 128x32 with 4 warps as an A6000-friendly head-dim-128 path,
  while newer SM8x tile-size code explores 128x128/8-warp variants for SM86. Sources:
  https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_launch_template.h
  and https://github.com/Dao-AILab/flash-attention/blob/main/hopper/tile_size.h

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_warp_row_diagnostic_guard.json`.
- The planner failed validation after three attempts.
- Final validation error: `candidate_patch is described as known invalid by the decision itself;
  found phrase 'must not be scored'`.
- No command ran beyond the internal `agent-plan` failure record, no patch applied, no score
  payload was produced, and lineage stayed unchanged.

Decision:

- Runtime retry feedback now has a targeted branch for `must not be scored`.
- The feedback says not to retry another compile-only skeleton with that language. It requires
  either an edit-mode raw diff complete enough to score after a successful compile, or a different
  no-edit diagnostic.
- Runtime knowledge records the planning failure and feedback adjustment.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 115 tests.
- `uv run --extra dev pytest`: passed, 205 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.40: Recorded env diagnostic guard

Success criteria for this checkpoint:

- Run one bounded loop after the must-not-score retry feedback.
- If the planner chooses a generic no-edit environment stability check, record the result and reject
  repeats unless there is a concrete environment/build failure.
- Preserve source and lineage.

Source refresh:

- Exa refreshed SM86/A6000 FlashAttention block-size context before this run. The useful details were
  that upstream FA2 block-size code uses smaller N tiles for SM8x/head-dim-128 cases and that newer
  tile-size logic has explicit `sm86_or_89` branches. Sources:
  https://github.com/Dao-AILab/flash-attention/blob/3387de49/flash_attn/flash_attn_interface.py
  and https://github.com/Dao-AILab/flash-attention/blob/main/hopper/tile_size.h

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_must_not_score_feedback.json`.
- The planner chose no-edit mode and ran `avo env`.
- The diagnostic passed and no source files, score payloads, or lineage entries changed.

Environment result:

- GPU target: NVIDIA RTX A6000, compute capability 8.6, `compute_86/sm_86`.
- PyTorch: `2.11.0+cu130`, CUDA available, torch CUDA `13.0`.
- NVCC: local cu13 package nvcc CUDA `13.0`, extension build compatibility `exact`.
- Baseline build environment: `CUDA_HOME`/`CUDA_PATH` point at the local cu13 package,
  `FLASH_ATTN_CUDA_ARCHS=80`, `MAX_JOBS=1`, `NVCC_THREADS=1`.
- Agent environment: Anthropic SDK installed, API key present.

Decision:

- Runtime repo context now states that the current CUDA/build environment is already recorded as
  stable and that no-edit `avo env` should not be used just to reconfirm stability.
- Runtime validation now rejects generic environment-stability `avo env` decisions unless the
  decision cites a concrete recent CUDA/build failure such as a version mismatch, missing compiler,
  missing package, or extension-build error.
- Runtime retry feedback explains how to correct that validation error.
- Runtime knowledge records the environment diagnostic.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 118 tests.
- `uv run --extra dev pytest`: passed, 208 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.41: Unsupported M=32 WMMA guard

Success criteria for this checkpoint:

- Run one bounded loop after the recorded env diagnostic guard.
- If the planner proposes a structural query-tile expansion, let the compile gate establish whether
  the WMMA fragment shapes are supported.
- If unsupported, preserve that fact in validation, runtime knowledge, and the lab notebook.

Source refresh:

- Exa refreshed Ampere async-copy and pipeline context before this run. The useful constraint was
  that real Ampere `cp.async` work needs aligned 16-byte global-to-shared groups, commit/wait groups,
  and a double-buffered shared-memory consumption path, matching the existing scalar-async-copy
  guard. Sources:
  https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/memory_sm80.h and
  https://docs.nvidia.com/cuda/ampere-tuning-guide/

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_env_guard.json`.
- The planner proposed changing `kTile` from 16 to 32 and using 32-row WMMA fragments such as
  accumulator/matrix_a/matrix_b `32x16x16`.
- The patch applied, NVCC rejected it, and cleanup reverse-applied it successfully.
- No score payload was produced and lineage stayed unchanged.

Compile failure:

- NVCC rejected `wmma::fragment<wmma::accumulator, 32, 16, 16, float>`.
- NVCC rejected BF16 `wmma::matrix_a` and `wmma::matrix_b` fragments with shape `32x16x16`.
- The failure mode was incomplete WMMA fragment types and no matching `fill_fragment` overload.

Decision:

- Runtime validation now rejects direct M=32 WMMA fragment probes, both literal
  `fragment<..., 32, 16, 16, ...>` and symbolic `kTile=32` plus `fragment<..., kTile, 16, 16, ...>`.
- Runtime knowledge records that this seed must stay on supported `16x16x16` WMMA fragments unless
  larger query tiles are implemented with multiple supported fragments or a different dataflow.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 120 tests.
- `uv run --extra dev pytest`: passed, 210 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.42: Self-described do-not-apply patch guard

Success criteria for this checkpoint:

- Run one bounded loop after the unsupported M=32 WMMA guard.
- If the planner identifies its own patch as malformed, reject that class before future patch
  application.
- Preserve the compile failure and validation fix in runtime knowledge and the experiment log.

Source refresh:

- Exa refreshed CUDA function-specifier context. NVIDIA's CUDA language docs define
  `__device__` and inlining specifiers such as `__forceinline__` as function annotations; this
  supports the local diagnosis that helper functions must be placed as complete declarations, not
  spliced into an existing `__global__` kernel parameter list.
  Source: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html

Loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_m32_wmma_guard.json`.
- The planner proposed adding warp-level reduction helpers above the MMA kernel, but the raw diff
  inserted those helpers after the first `__global__ void mma_attention_kernel(` line and before the
  existing parameter list.
- The patch applied, NVCC rejected it, and cleanup reverse-applied it successfully.
- No score payload was produced and lineage stayed unchanged.

Compile failure:

- NVCC reported `invalid specifier on a parameter` at the inserted
  `__device__ __forceinline__` helper.
- The next parse errors came from the helper body being located inside the kernel declaration
  context instead of at file scope.
- The decision risk text already said this would cause NVCC to fail and explicitly said
  "Do not apply this patch".

Decision:

- Runtime validation now treats `do not apply this patch` and `will cause nvcc to fail` as
  self-invalid descriptions for non-empty candidate patches.
- Runtime knowledge records that CUDA helper functions must be placed as complete file-scope
  declarations before the kernel declaration or after the kernel body, not inside a function
  signature.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 121 tests.
- `uv run --extra dev pytest`: passed, 211 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.43: Transform and failure-class redesign

Success criteria for this checkpoint:

- Stop adding one-off CUDA guardrails for every malformed attempt.
- Add a structured edit channel so the planner can request tiny transformations instead of raw CUDA
  diffs.
- Add generalized attempt failure classes so recurring failures can be promoted to hard preflight
  tracks by class, rather than by accumulating phrase bans.

Source refresh:

- Exa refreshed agentic CUDA-kernel tuning context. The relevant design signal was to constrain
  edits to localized, incremental transformations and let the correctness/performance harness decide
  whether the change persists, instead of letting the model freely regenerate brittle patches.
  Sources: https://www.arxiv.org/pdf/2603.21331 and
  https://arxiv.org/html/2601.12698v1

Design change:

- `VariationDecision` now accepts an optional `candidate_transform` object, mutually exclusive with
  raw `candidate_patch`.
- Supported transform ops are `replace_once`, `insert_before_once`, `insert_after_once`, and
  `set_constexpr_int`.
- `run_decision_command` materializes a transform into a normal candidate patch, applies the same
  candidate-only patch safety checks, records the generated patch in the attempt, and still uses the
  existing cleanup path for rejected candidates.
- Raw unified diffs remain as a legacy fallback, but the prompt and repo context now steer CUDA
  edits toward structured transforms.

Failure classification:

- The self-invalid phrase list was replaced with planning-risk classes:
  no-effect/skeleton, incomplete or malformed edit, predicted compile failure, and predicted
  correctness failure.
- Attempt summaries now include a generalized failure class such as raw-diff preflight, structured
  transform preflight, patch-safety preflight, unsupported WMMA shape, CUDA syntax error, stale or
  undefined symbol, correctness failure, throughput regression, or compile-only diagnostic.
- Supervisor history now detects repeated unaccepted failure classes and asks the planner to promote
  that class to a hard preflight track or switch transform families.

Decision:

- This is a reliability redesign, not a kernel-performance checkpoint. No new score was accepted and
  the lineage head stays at the direct-accumulation MMA kernel.
- The next evolution step should use `candidate_transform` when the edit is expressible as a small
  replace/insert/constexpr change. Recurring CUDA mistakes should become generalized preflight
  classes, not single-attempt ban phrases.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 124 tests.
- `uv run --extra dev pytest tests/test_evolve.py -q`: passed, 41 tests.
- `uv run --extra dev pytest`: passed, 218 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.44: Transform-mode preflight tightening

Success criteria for this checkpoint:

- Run one bounded live loop after the transform/failure-class redesign.
- If the live planner exposes a schema or mode-consistency gap, close that gap generally instead of
  adding a CUDA-specific one-off guard.
- Verify the fix with focused tests and the full suite.

Source refresh:

- Exa refreshed constrained CUDA-kernel tuning work before the live loop. CuTeGen-style staged
  repair emphasizes first diagnosing failures, then applying localized structured edits instead of
  regenerating broad patches. Source: https://www.arxiv.org/pdf/2604.01489

Live loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_transform_planning.json`.
- The planner returned `candidate_edit` beginning with `No edit;` but also supplied a
  `candidate_transform`.
- The transform path was malformed with whitespace-separated tokens:
  `candidates / cud a_w arp _ rows _ attention _kernel . cu`.
- Execution rejected the transform before running the score command, so no CUDA kernel launched, no
  score payload was produced, and lineage stayed unchanged.

Decision:

- Runtime decision parsing now rejects malformed `candidate_transform.path` values before
  execution. Transform paths must be repo-relative candidate source paths under `candidates/`, use
  a source-file suffix, and contain no whitespace, backslashes, traversal, absolute path, or NUL.
- Runtime decision parsing now rejects `No edit;` decisions that include any edit payload, whether
  `candidate_transform` or raw `candidate_patch`.
- This is a mode-preflight fix for the structured interface, not a CUDA-kernel guard.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 126 tests.
- `uv run --extra dev pytest`: passed, 220 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.45: Transform-channel retry feedback

Success criteria for this checkpoint:

- Run a follow-up bounded live loop after transform-mode preflight tightening.
- If planning fails because the agent still confuses edit channels, improve retry feedback rather
  than adding another CUDA-domain guard.
- Verify with focused tests and full checks.

Source refresh:

- Exa refreshed structured-output retry patterns. The useful reliability pattern is to feed precise
  validation errors back to the model so schema relationship failures are corrected on retry, rather
  than accepting partial objects downstream. Source:
  https://enricopiovano.com/blog/structured-outputs-tool-use-patterns

Live loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_transform_preflight_tightening.json`.
- Planning failed after three retries before any candidate command executed.
- The final validation error was `candidate_patch and candidate_transform are mutually exclusive`;
  the planner kept mixing the new structured transform channel with the legacy raw diff channel.
- No score payload was produced and lineage stayed unchanged.

Decision:

- Retry feedback now has a dedicated branch for mutually exclusive edit channels.
- The prompt now says structured-transform mode requires `candidate_patch` to be exactly the empty
  string `""`.
- Retry feedback also has a dedicated branch for `No edit;` decisions that carry edit payloads.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 128 tests.
- `uv run --extra dev pytest`: passed, 222 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.46: Transform-aware score-repeat feedback

Success criteria for this checkpoint:

- Run a live loop after transform-channel retry feedback.
- If old validation messages still steer the planner toward raw diffs only, update them to advertise
  the structured transform channel consistently.
- Verify with focused tests and full checks.

Live loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_transform_channel_feedback.json`.
- Planning failed after three retries before any candidate command executed.
- The final validation error was a recorded unpatched MMA seed score repeat. This means the edit
  channel confusion was resolved, but the remaining score-repeat guidance still said only
  `candidate_patch`.
- No score payload was produced and lineage stayed unchanged.

Decision:

- Recorded seed-score repeat errors now say `candidate_transform/candidate_patch` instead of only
  `candidate_patch`.
- Retry feedback for unpatched MMA and warp-row seed scores now recommends `candidate_transform`
  or a legacy raw patch.
- No CUDA-domain ban was added.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 128 tests.
- `uv run --extra dev pytest`: passed, 222 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.47: Raw CUDA patch channel closed

Success criteria for this checkpoint:

- Run a live loop after transform-aware score-repeat feedback.
- If the planner still uses a raw CUDA diff for kernel evolution, close that interface path and keep
  raw diffs only for non-CUDA candidate files.
- Record the score result if the patch is executable.

Live loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_transform_score_feedback.json`.
- The planner produced a valid legacy raw diff against
  `candidates/cuda_warp_rows_attention/attention_kernel.cu`.
- The patch removed shared `v_tiles` staging and loaded V directly from global memory in the
  warp-row PV accumulation loop.
- The score passed correctness but gate-rejected throughput:
  geomean `0.40530677363112055` TFLOPS versus best `0.5772885607891738`.
- Noncausal: max error `0.001953125`, median `0.9009280204772949 ms`,
  `0.5959087738391973` TFLOPS.
- Causal: max error `0.015625`, median `0.9737600088119507 ms`,
  `0.27566900834992025` TFLOPS.
- Cleanup reverse-applied the patch successfully and lineage stayed unchanged.

Decision:

- Runtime validation now rejects raw `candidate_patch` edits to `.cu` and `.cuh` files after
  existing domain-specific preflights run.
- CUDA kernel edits must use `candidate_transform`, which is materialized and checked by the
  orchestrator. Raw diffs remain available for non-CUDA candidate files such as Python wrappers.
- Retry feedback and prompt/context text now explain that raw CUDA patches are not an allowed edit
  channel.

Verification:

- `uv run --extra dev pytest tests/test_agent.py -q`: passed, 130 tests.
- `uv run --extra dev pytest`: passed, 224 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.48: Structural transform preflight and promotion

Success criteria for this checkpoint:

- Run a live loop after closing the raw CUDA patch channel.
- If planning fails because the planner describes a CUDA edit but provides neither transform nor
  patch, make the validation feedback point at the structured transform channel instead of the old
  patch-only wording.
- Replace remaining reactive CUDA phrase guards with named structural preflight tracks.
- Persist recurring failure-class promotions instead of only telling the planner to promote them.
- Recenter guidance on realistic long-sequence target workloads, with small candidate shapes treated
  as smoke-only fences.
- Verify with focused tests and full checks.

Live loop result:

- Command: `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage
  --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir ./attempts
  --max-steps 1 --timeout-s 300 --loop-json attempts/loop_after_raw_cuda_patch_closure.json`.
- Planning failed after three retries before any candidate command executed.
- The final invalid decision described adding a `probability_frag_next` WMMA preload skeleton but
  supplied no `candidate_transform` and no raw patch.
- No score payload was produced and lineage stayed unchanged.
- A follow-up live loop after structural preflight/promotion changes still failed planning after
  three retries. The planner described a `candidate_transform set_constexpr_int` change to
  `kMaxSeqLen` but omitted the transform object.
- A later live loop after explicit transform-shorthand recovery still failed planning after three
  retries with a recorded no-patch compile diagnostic. The failure is now classified as a planning
  validation class instead of `unknown`.

Decision:

- The validation error for code-change descriptions without an edit payload now says
  `candidate_transform or candidate_patch must be provided`.
- Retry feedback still offers the exact three modes, with structured transform first and raw patch
  only as legacy non-CUDA fallback.
- Runtime CUDA patch validation is now expressed as named structural preflight tracks:
  edit-channel integrity, WMMA fragment shape/type, async-copy granularity/API shape, tile-local
  shared-memory addressing, symbol lifecycle, complete shape graduation, and no-effect skeletons.
- Materialized `candidate_transform` patches run structural preflight before git apply and before
  compile/score execution.
- Evolve-loop promotion is now persistent. Recurring promotable failure classes are written to
  `attempts/preflight_tracks.json`, summarized as active hard preflight tracks, and loaded back into
  command execution.
- Prompt/context guidance no longer lists many historical "do not repeat X" patches. It points the
  planner at transform families and structural failure classes, and names the realistic BF16 target
  workload family: seq 4096/8192/16384/32768, total_tokens 32768, num_heads 16, head_dim 128, both
  causal modes.
- The parser now recovers explicit single-constant prose transforms like "change NAME from OLD to
  NEW in candidates/.../*.cu" into `candidate_transform={op: set_constexpr_int, path, name, value}`.
- Planning validation failures are classified into durable classes such as
  `planning_missing_edit_payload`, `planning_no_patch_compile`, and `planning_edit_channel`, so
  repeated planner-interface failures can promote like compile/runtime failures.

Verification:

- `uv run --extra dev pytest tests/test_agent.py tests/test_evolve.py -q`: passed, 176 tests.
- `uv run --extra dev pytest`: passed, 229 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.49: Batched transforms and compile-loop preflight

Success criteria for this checkpoint:

- Fix the false repeated-fingerprint signal caused by synthetic planning-failure records.
- Make shape-graduation edits expressible as a small structured batch, not raw CUDA diffs.
- Keep batch transforms generic enough to avoid another `SMOKE_SEQUENCES`-only recovery path.
- Prevent the planner from spending repeated loop steps on the same successful compile-only
  structured transform.
- Verify that the loop can graduate beyond the seq256 smoke cap and score a larger validation
  workload.

Research note:

- Exa search for agent retry/failure-loop design surfaced the same principle in multiple sources:
  classify failures before deciding retry or fallback behavior, persist failure signatures to avoid
  infinite implement-verify loops, and enforce deterministic guardrails around probabilistic planner
  output.
- Useful references:
  https://clarion.ai/insights-resilient-agentic-ai-pipelines-retry-fallback-human-in-the-loop/,
  https://docs.fabro.sh/execution/failures, and
  https://agentpatterns.ai/verification/deterministic-guardrails/.

Live loop results:

- `attempts/loop_after_planning_fingerprint_fix.json`: the planner inferred a
  `set_constexpr_int` transform for `kMaxSeqLen=512`, applied it, and compiled successfully on
  sm86 with no spills, 40 registers, 1 barrier, and 9920 bytes shared memory. It produced no score
  payload and cleanup reverted the patch.
- Anthropic tool-schema iterations showed that nested `steps` arrays made the provider schema too
  complex. The provider-facing batch interface is now `op=batch` plus compact `steps_json`; local
  validation still normalizes it into `steps`.
- `attempts/loop_after_python_set_transform.json` and
  `attempts/loop_after_batch_score_validator.json`: the planner materialized the two-file batch
  transform (`kMaxSeqLen=512` plus adding `512` to the wrapper sequence set) and compile-checked it
  successfully, but kept repeating compile-only diagnostics.
- `attempts/loop_after_repeated_compile_hard_preflight.json`: after adding the cross-step hard
  preflight against repeated successful compile-only transforms, the planner scored the same batch
  instead of compiling again.
- The seq512 score passed correctness:
  - Noncausal: max error `0.001953125`, median `1.3272960186004639 ms`,
    `3.2358774800882233` TFLOPS.
  - Causal: max error `0.0078125`, median `1.2746880054473877 ms`,
    `1.6847131524127585` TFLOPS.
  - Geomean: `2.334850177270671` TFLOPS on `seq_len=512`, `total_tokens=2048`,
    `num_heads=8`, `head_dim=128`, BF16, both causal modes.
- Lineage stayed unchanged because the gate compares against the current best shape set and rejected
  the candidate as `candidate benchmark cases differ from current best`.

Decision:

- Attempt fingerprints now include planning failure class and truncated validation detail for
  planning-validation failures, so distinct planner-interface errors do not collapse into one bogus
  fingerprint.
- `candidate_transform` now supports a small ordered batch of tiny steps. The tool schema uses
  `steps_json` to stay provider-compatible; runtime validation enforces one to four step objects.
- Added generic `add_int_to_python_set` transform materialization. It updates a single-line Python
  integer set assignment and rejects ambiguous or non-integer set contents.
- Parser recovery can infer generic Python set names such as `ALLOWED_SEQUENCES` from prose and
  `files_to_inspect`, rather than hardcoding the historical `SMOKE_SEQUENCES` case.
- Removed unused exact CUDA attempt-pattern helpers from `agent.py`; the remaining hard checks are
  structural tracks such as WMMA shape/type, async-copy granularity/API shape, symbol lifecycle,
  tile-local shared-memory addressing, complete shape graduation, and no-effect skeletons.
- Patched MMA scores beyond the smoke cap are now allowed only when the structured transform changes
  both the wrapper and kernel paths together. Wrapper-only cap edits are rejected.
- Attempt history now emits a follow-up signal after a successful compile-only structured transform,
  and command planning has a hard cross-step preflight that rejects repeating that same transform
  with another compile command. The loop must score it or choose a materially different transform
  family.

Verification:

- `uv run --extra dev pytest tests/test_agent.py tests/test_evolve.py -q`: passed, 187 tests.
- `uv run --extra dev pytest`: passed, 240 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

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

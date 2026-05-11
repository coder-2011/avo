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

## 2026-05-08 - Checkpoint 4.50: Suite-aware lineage lanes

Success criteria for this checkpoint:

- Unblock migration from tiny smoke suites to larger validation workloads without comparing
  unrelated benchmark signatures.
- Preserve apples-to-apples throughput gating within each benchmark case signature.
- Record the previously scored seq512 MMA shape-graduation result as a lineage lane with source and
  patch artifacts.

Problem:

- The seq512 structured batch score from checkpoint 4.49 passed correctness and reached
  `2.334850177270671` geomean TFLOPS, but the lineage gate rejected it because the latest accepted
  score was the smaller seq256/head_dim128 smoke suite.
- That rejection preserved apples-to-apples comparison, but it also trapped the search on the old
  smoke suite and prevented larger validation workloads from becoming durable lineage evidence.

Decision:

- Runtime lineage gating now looks up the best prior payload with the same benchmark case signature.
- A candidate that matches an existing signature must match or improve that signature's best
  geomean, preserving apples-to-apples gating.
- A candidate with a new valid signature can establish a new benchmark case lane when it passes
  correctness and has a finite positive geomean.
- Accepted scores are written to `scores/by_signature/<hash>.json` as well as `scores/latest.json`.
- The runtime knowledge base and README now describe the suite-aware gate.

Lineage result:

- Replayed the already-recorded seq512 score through the new gate by applying the recorded batch
  patch, snapshotting the patched wrapper plus companion CUDA source directory, committing the score
  to the nested lineage repo, and reverting the runtime worktree patch.
- Nested lineage commit: `bc05dd85ccc8872308c786ba57664cfdffdc4640`
  (`evolve: accept mma seq512 shape lane`).
- Gate decision: accepted, best geomean `0.0`, candidate geomean `2.334850177270671`, reason
  `candidate established benchmark case set`.
- The nested lineage source snapshot includes `candidates/cuda_mma_attention_seed.py`,
  `candidates/cuda_mma_attention/attention.cpp`, and
  `candidates/cuda_mma_attention/attention_kernel.cu`.

Verification:

- `uv run --extra dev pytest tests/test_lineage.py tests/test_evolve.py -q`: passed, 61 tests.
- `uv run --extra dev pytest`: passed, 241 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.51: Lineage lane summaries and cap-consistent scoring

Success criteria for this checkpoint:

- Ensure the agent sees all accepted benchmark lanes, not only `scores/latest.json`.
- Keep the "score this compiled transform" follow-up signal alive across planning failures.
- Prevent structured shape-graduation scores from requesting sequence lengths outside the cap that
  the same transform expressed.
- Run at least one live loop to validate the prompt/supervisor behavior after the lineage summary
  change.

Live loop results:

- `attempts/loop_after_lineage_lane_summary.json`: with the multi-lane lineage summary, the planner
  proposed a new seq1024 MMA shape-graduation batch (`kMaxSeqLen=1024`, add `1024` to
  `SMOKE_SEQUENCES`). Compile passed on sm86 with no spills, 40 registers, 1 barrier, and 9920 bytes
  shared memory. No score payload was produced, and cleanup reverted the patch.
- `attempts/loop_after_1024_compile_followup.json`: planning failed before execution because the
  planner returned a repeated no-patch compile diagnostic after retries.
- `attempts/loop_after_persistent_compile_followup.json`: after making the compile-followup signal
  persist across planning failures, the planner scored the seq1024 batch instead of compiling again.
  The score command requested `seq_lens=1024,2048,4096`, `total_tokens=32768`, `num_heads=16`,
  `head_dim=128`, BF16, both causal modes.
- Seq1024 cases passed correctness:
  - Noncausal: max error `0.001953125`, median `27.58780860900879 ms`,
    `9.963745610959426` TFLOPS.
  - Causal: max error `0.0078125`, median `27.701984405517578 ms`,
    `4.961339644846` TFLOPS.
- Seq2048 and seq4096 cases failed wrapper validation because the transform only added the seq1024
  cap. The gate rejected the score for correctness failure.

Decision:

- `lineage_score_summary()` now reports the latest accepted payload and the best payload for each
  benchmark-signature lane. The CLI uses that summary for agent prompts.
- New benchmark signatures can be committed for unchanged source snapshots; same-signature unchanged
  source reruns still reject as timing-noise probes.
- Attempt history now keeps a pending compile-only structured transform follow-up active until that
  exact transform is scored, even if planning failures occur after the compile.
- MMA score validation now checks that requested `seq_lens` are covered by the structured transform:
  `kMaxSeqLen` must be high enough, and the wrapper sequence set must include every non-base smoke
  sequence requested by the score command.

Research note:

- Exa search on coding-agent memory and context compaction reinforced that durable, compact
  summaries should preserve synthesized state rather than raw interaction logs. The lineage-lane
  summary follows that pattern: it gives the planner the accepted score state needed for decisions
  without injecting full git history or large score payloads.

Verification:

- `uv run --extra dev pytest tests/test_lineage.py tests/test_cli.py -q`: passed, 35 tests.
- `uv run --extra dev pytest tests/test_evolve.py tests/test_lineage.py -q`: passed, 65 tests.
- `uv run --extra dev pytest tests/test_agent.py tests/test_evolve.py tests/test_lineage.py -q`:
  passed, 205 tests.
- `uv run --extra dev pytest`: passed, 247 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.52: Promoted structural preflights and seq1024 source lane

Success criteria for this checkpoint:

- Address the critique that prompt/guard work was still too reactive and tied to exact prior
  failures.
- Make recurring failure-class promotion affect hard preflight behavior, not just prompt advice.
- Move no-edit MMA scoring away from tiny smoke workloads now that a seq1024 lane exists.
- Preserve the accepted seq1024 runtime source change and verify the full suite.

Live loop context:

- `attempts/loop_after_transform_cap_validator.json`: the planner described extending
  `kMaxSeqLen` from 256 to 1024 but omitted the edit payload, producing a planning-validation
  failure.
- `attempts/loop_after_extend_constant_recovery.json`: parser recovery inferred the tiny constant
  transform and the planner produced a valid seq1024 batch compile. Compile passed on sm86 with no
  spills, 40 registers, 1 barrier, and 9920 bytes shared memory.
- `attempts/loop_after_repeated_compile_block.json`: after repeated-compile blocking, the planner
  moved toward a larger cap but made the wrapper/kernel sequence caps inconsistent.
- `attempts/loop_after_mma_compile_cap_consistency.json`: after cap-consistency validation, the
  planner scored the prior valid seq1024 batch. The nested lineage gate accepted it as commit
  `a9f55492223c977ea4525347145fe991f5e06174`.

Seq1024 accepted lane:

- Workload: `seq_len=1024`, `total_tokens=8192`, `num_heads=8`, `head_dim=128`, BF16, both causal
  modes.
- Noncausal: max error `0.00390625`, median `4.917280197143555 ms`,
  `6.987549415621984` TFLOPS.
- Causal: max error `0.0078125`, median `4.663455963134766 ms`,
  `3.6839351158902605` TFLOPS.
- Geomean: `5.073625790914057` TFLOPS.
- Runtime source now preserves that lane directly: `kMaxSeqLen=1024` in the MMA CUDA source and
  `1024` in the wrapper `SMOKE_SEQUENCES`.

Decision:

- Planning-risk text checks were generalized around failure classes such as no-effect skeletons,
  incomplete/malformed edits, stale symbols, predicted compile failure, and predicted correctness
  failure. Exact historical phrases such as one-off duplicate-line notes were removed from the
  prompt-facing classifier.
- The repo context now describes no-patch compiles as baseline diagnostics and describes the MMA
  source as accepted through seq1024, rather than embedding a list of historical "do not repeat X"
  notes.
- Active promoted failure classes now flow into materialized transform preflight. For promoted
  `stale_or_undefined_symbol`, the hard track rejects CUDA edits that remove a declaration while
  still adding uses of the old identifier, and rejects duplicate local declarations in one edit.
- WMMA fragment-shape preflight now parses added `wmma::fragment<...>` templates and resolves
  added `constexpr int` dimensions, rejecting resolvable dimensions outside Ampere BF16 16x16x16
  support. This replaces narrower checks for specific prior k32/m32 spellings.
- Sequence-cap graduation is enforced structurally: increasing `kMaxSeqLen` must carry the matching
  `SMOKE_SEQUENCES` wrapper cap in the same batch, and wrapper-only additions beyond the accepted
  base cap are rejected.
- No-edit MMA candidate scores below the accepted seq1024 lane are rejected. Future scoring should
  start at seq1024 or use a structured wrapper/kernel shape-graduation batch toward realistic long
  sequences.
- Repeating a previously successful compile-only transform is blocked even after that transform has
  been scored, so the planner cannot return to compile-only loops after a failed score.

Verification:

- `uv run --extra dev pytest tests/test_agent.py tests/test_evolve.py -q`: passed, 199 tests.
- `uv run --extra dev pytest`: passed, 255 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.53: Repair MMA wrapper/kernel sequence guard drift

Success criteria for this checkpoint:

- Fix an executable source mismatch introduced by preserving the accepted seq1024 source lane.
- Add a low-cost regression test so the wrapper's explicit MMA sequence set cannot drift away from
  the CUDA guard again.
- Refresh the runtime README and knowledge notes so they reflect the accepted seq1024 lane instead
  of the older 256-token-only state.

Problem:

- `candidates/cuda_mma_attention_seed.py` advertised
  `SMOKE_SEQUENCES = {16, 32, 64, 128, 256, 1024}`.
- After `kMaxSeqLen` was changed to `1024`, the CUDA `TORCH_CHECK` still accepted only
  `16/32/64/128/kMaxSeqLen`. That meant wrapper-accepted seq256 inputs would fail inside the
  extension even though seq256 remains part of the local smoke lane history.
- The same shape would have hurt future shape graduation: if `kMaxSeqLen` later became 4096, the
  kernel would only accept 4096 plus tiny base shapes, not intermediate wrapper-advertised lanes
  such as 1024 or 2048.

Decision:

- Changed the CUDA guard to accept any multiple of `kTile` up to `kMaxSeqLen`.
- Added `test_mma_wrapper_sequences_are_supported_by_kernel_guard`, which parses the wrapper's
  `SMOKE_SEQUENCES`, parses the CUDA `kMaxSeqLen`, checks that wrapper sequences are multiples of
  16 and at or below the cap, and asserts the CUDA guard uses `seq_len <= kMaxSeqLen` plus
  `seq_len % kTile == 0`.
- Updated the runtime README to describe the current state as an accepted seq1024 BF16 WMMA lane,
  while keeping the "not faster than FlashAttention-2 yet" caveat explicit.
- Updated the Ampere knowledge base with the source-drift fix and the direct seq256 regression
  score after the change.

Regression score:

- Command:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_mma_attention_seed.py --seq-lens 256 --total-tokens 1024 --num-heads 4 --head-dim 128 --dtype bf16 --causal both --repeats 1 --warmup 0 --timeout-s 300`
- Result: correctness passed for both cases.
- Noncausal: max error `0.001953125`, median `0.6199679970741272 ms`,
  `0.8659655248879056` TFLOPS.
- Causal: max error `0.0078125`, median `0.8414720296859741 ms`,
  `0.3190069860078135` TFLOPS.
- Geomean: `0.525593999281922` TFLOPS.

Research note:

- Exa refresh linked the next search direction back to primary Ampere/FA2 evidence:
  FlashAttention-2 emphasizes avoiding split-K-style shared-memory traffic by splitting Q across
  warps, while NVIDIA/CUTLASS Ampere references reinforce 16-byte `cp.async` groups and BF16
  `mma.sync.aligned.m16n8k16` as the relevant sm86 primitive family. Sources:
  https://arxiv.org/abs/2307.08691,
  https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html,
  https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm80.hpp, and
  https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/memory_sm80.h.

Verification:

- `uv run --extra dev pytest tests/test_candidate_backend.py -q`: passed, 5 tests.
- `uv run --extra dev pytest`: passed, 256 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.54: Bounded loop accepts seq2048 MMA lane

Success criteria for this checkpoint:

- Run a bounded autonomous loop after the sequence-guard repair.
- Commit any accepted lane into the nested lineage and preserve the accepted source patch in the
  runtime repo.
- Update validation constants and repo notes so the agent treats seq2048 as the current accepted
  lane rather than seq1024.

Live loop:

- Command:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 3 --loop-json attempts/loop_after_sequence_guard_repair.json --timeout-s 900 --env-file ../avo/.env.local`
- Step 1: planner emitted a two-step structured batch (`kMaxSeqLen=2048`, add `2048` to
  `SMOKE_SEQUENCES`) and compile-checked it. Compile passed with no spills, 40 registers, 1 barrier,
  and 9920 bytes shared memory. The rejected compile-only patch was cleaned up.
- Step 2: planner returned an invalid missing-payload decision for the same shape graduation; this
  was recorded as a planning-validation failure.
- Step 3: planner followed the compile signal and scored the same seq2048 batch. The score passed
  correctness and the suite-aware gate accepted a new benchmark lane.

Accepted seq2048 lane:

- Nested lineage commit: `39af063` (`evolve: accept candidate`).
- Workload: `seq_len=2048`, `total_tokens=16384`, `num_heads=16`, `head_dim=128`, BF16, both causal
  modes.
- Noncausal: max error `0.001953125`, median `27.609344482421875 ms`,
  `9.955973678368473` TFLOPS.
- Causal: max error `0.0078125`, median `27.304447174072266 ms`,
  `5.033573930129197` TFLOPS.
- Geomean: `7.079133390217197` TFLOPS.
- Gate decision: accepted with reason `candidate established benchmark case set`.

Decision:

- Runtime source now carries `kMaxSeqLen=2048` and wrapper
  `SMOKE_SEQUENCES={16, 32, 64, 128, 256, 1024, 2048}`.
- Agent-side current MMA base sequences now include 2048, and no-edit MMA scores below seq2048 are
  treated as below the accepted validation lane.
- Patched MMA score validation now checks transformed caps before accepting a score as an in-base
  smoke shape, so a transform that lowers or narrows `kMaxSeqLen` cannot score a larger current-base
  sequence by accident.
- README and knowledge notes now describe seq2048 as the current accepted MMA lane.

Verification:

- The loop itself compiled and scored the accepted candidate successfully on A6000/sm86.
- `uv run --extra dev pytest`: passed, 258 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.55: Exact pending-transform follow-up and seq4096 lane

Success criteria for this checkpoint:

- Fix the planner-interface failure where a compiled structured transform is referenced in prose
  but omitted from the follow-up score payload.
- Re-run a bounded loop from the seq2048 source and try to establish the first target-shape lane
  at seq4096.
- Preserve any accepted seq4096 source state and update validation constants, tests, README, and
  knowledge notes.

Problem:

- `attempts/loop_after_seq2048_lane.json` did compile-check the seq4096 structured batch
  (`kMaxSeqLen=4096`, add `4096` to `SMOKE_SEQUENCES`) successfully, with no spills, 40 registers,
  1 barrier, and 9920 bytes shared memory.
- The next planner turn then tried to score "the compiled seq4096 transform" while omitting the
  exact `candidate_transform`, so validation rejected it as a missing edit payload. The bounded
  loop stopped without a score.

Decision:

- Attempt-history follow-up signals now include the exact compact pending `candidate_transform`
  JSON. This gives the planner a concrete object to reuse instead of paraphrasing the transform.
- Validation feedback for missing edit payloads now explicitly says that a follow-up score for a
  compiled transform must include the exact `candidate_transform` object from the follow-up signal.
- Added tests that attempt history includes the pending transform JSON even after an intervening
  planning failure, plus feedback coverage for the new guidance.

Accepted seq4096 lane:

- Command:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 3 --loop-json attempts/loop_after_pending_transform_json.json --timeout-s 900 --env-file ../avo/.env.local`
- Nested lineage commit: `7162462` (`evolve: accept candidate`).
- Workload: `seq_len=4096`, `total_tokens=32768`, `num_heads=16`, `head_dim=128`, BF16, both causal
  modes, `trials=3`, `warmup=1`, `repeats=1`.
- Noncausal: max error `0.0009765625`, median `103.79443359375 ms`,
  `10.59316564199843` TFLOPS, timing CV `0.013159526868716758`.
- Causal: max error `0.0078125`, median `102.7823715209961 ms`,
  `5.348736419996861` TFLOPS, timing CV `0.0031471738936156303`.
- Geomean: `7.527287085824243` TFLOPS.
- Gate decision: accepted with reason `candidate established benchmark case set`.

Decision:

- Runtime source now carries `kMaxSeqLen=4096` and wrapper
  `SMOKE_SEQUENCES={16, 32, 64, 128, 256, 1024, 2048, 4096}`.
- Agent-side current MMA base sequences now include 4096, no-edit MMA scores below seq4096 are
  treated as below the accepted validation lane, and the exact seq4096 no-edit score is recorded.
- README and knowledge notes now describe seq4096 as the current accepted MMA lane.

Verification:

- `uv run --extra dev pytest tests/test_evolve.py::test_summarize_attempt_history_requests_score_after_compile_only_transform tests/test_evolve.py::test_summarize_attempt_history_keeps_compile_followup_after_planning_failure tests/test_agent.py::test_decision_feedback_explains_empty_patch_validation_error -q`: passed, 3 tests.
- `uv run --extra dev pytest`: passed, 256 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.56: Bounded loop accepts seq8192 MMA lane

Success criteria for this checkpoint:

- Continue the shape-graduation path from the accepted seq4096 lane toward the remaining long target
  shapes.
- Preserve any accepted seq8192 source state and update validation constants, tests, README, and
  knowledge notes.
- Keep correctness and timing evidence attached to the accepted lane.

Live loop:

- Command:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 3 --loop-json attempts/loop_after_seq4096_lane.json --timeout-s 900 --env-file ../avo/.env.local`
- Step 1: planner emitted a structured seq8192 cap batch (`kMaxSeqLen=8192`, add `8192` to
  `SMOKE_SEQUENCES`) and compile-checked it. Compile passed with no spills, 40 registers, 1 barrier,
  and 9920 bytes shared memory.
- Step 2: planner reused the pending transform JSON and scored the seq8192 lane. Correctness passed
  and the suite-aware gate accepted a new benchmark lane.

Accepted seq8192 lane:

- Nested lineage commit: `b6622ab` (`evolve: accept candidate`).
- Workload: `seq_len=8192`, `total_tokens=32768`, `num_heads=16`, `head_dim=128`, BF16, both causal
  modes, `trials=3`, `warmup=1`, `repeats=1`.
- Noncausal: max error `0.0009765625`, median `211.3244171142578 ms`,
  `10.405911846727314` TFLOPS, timing CV `0.002891365967579352`.
- Causal: max error `0.0078125`, median `210.7821502685547 ms`,
  `5.216341262175792` TFLOPS, timing CV `0.0013814822341183689`.
- Geomean: `7.367549615485977` TFLOPS.
- Gate decision: accepted with reason `candidate established benchmark case set`.

Decision:

- Runtime source now carries `kMaxSeqLen=8192` and wrapper
  `SMOKE_SEQUENCES={16, 32, 64, 128, 256, 1024, 2048, 4096, 8192}`.
- Agent-side current MMA base sequences now include 8192, no-edit MMA scores below seq8192 are
  treated as below the accepted validation lane, and the exact seq8192 no-edit score is recorded.
- README and knowledge notes now describe seq8192 as the current accepted MMA lane.
- This is correctness and lane-establishment progress; it still does not satisfy the final
  objective of beating FlashAttention-2 on the target suite.

Verification:

- The loop itself compiled and scored the accepted candidate successfully on A6000/sm86.
- `uv run --extra dev pytest`: passed, 256 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.57: Promote recurring failure classes by tail counts

Success criteria for this checkpoint:

- Reduce the remaining whack-a-mole behavior in attempt memory.
- Promote recurring CUDA failure classes to hard preflight state even when the failures are
  interleaved, not only when the last three attempts are the exact same class.
- Keep the change structural and class-based instead of adding another historical phrase ban.

Decision:

- `update_promoted_preflight_tracks` now scans the unaccepted attempt tail since the last accepted
  result, counts classified failure classes, and persists every promotable class that reaches the
  repeat threshold.
- Supervisor summaries now report recurring classes with counts, for example
  `cuda_syntax_error(count=3)` and `stale_or_undefined_symbol(count=3)`.
- The persisted `preflight_tracks.json` state is still loaded before materialized transform/patch
  preflight, so promoted classes affect the next bounded step rather than remaining prompt-only
  advice.
- The legacy `domain_sanity` helper name was removed in favor of the structural preflight path.

Verification:

- `uv run --extra dev pytest tests/test_evolve.py::test_summarize_attempt_history_promotes_recurring_failure_class tests/test_evolve.py::test_update_promoted_preflight_tracks_persists_recurring_class tests/test_evolve.py::test_summarize_attempt_history_counts_mixed_recurring_failure_classes tests/test_evolve.py::test_update_promoted_preflight_tracks_persists_mixed_recurring_classes -q`: passed, 4 tests.
- `uv run --extra dev ruff check avo/evolve.py tests/test_evolve.py avo/agent.py`: passed.
- `uv run --extra dev pytest`: passed, 258 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.58: Bounded loop accepts seq16384 MMA lane

Success criteria for this checkpoint:

- Continue from the accepted seq8192 lane toward the remaining long target shapes.
- Preserve the accepted seq16384 source state and update validation constants, tests, README, and
  knowledge notes.
- Keep correctness and timing evidence attached to the accepted lane.

Research context:

- Exa search re-confirmed two relevant Ampere references before the run:
  NVIDIA CUTLASS's Ampere FA2-style example uses 128-thread CTA kernels with 128x128-style tiles
  and `cp.async`; FlashAttention-2's sm8x head_dim128 heuristic uses smaller N-block choices on
  sm86, with causal/no-dropout differing from noncausal. This remains search-space evidence rather
  than a direct instruction for the current minimal shape-graduation step.

Live loop:

- Command:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 3 --loop-json attempts/loop_after_seq8192_lane.json --timeout-s 900 --env-file ../avo/.env.local`
- Step 1: planner emitted a structured seq16384 cap batch (`kMaxSeqLen=16384`, add `16384` to
  `SMOKE_SEQUENCES`) and compile-checked it. Compile passed with no spills, 40 registers, 1 barrier,
  and 9920 bytes shared memory.
- Step 2: planner reused the pending transform JSON and scored the seq16384 lane. Correctness
  passed and the suite-aware gate accepted a new benchmark lane.

Accepted seq16384 lane:

- Nested lineage commit: `47ddfcf` (`evolve: accept candidate`).
- Workload: `seq_len=16384`, `total_tokens=32768`, `num_heads=16`, `head_dim=128`, BF16, both
  causal modes, `trials=3`, `warmup=1`, `repeats=1`.
- Noncausal: max error `0.00048828125`, median `436.07733154296875 ms`,
  `10.08547382076117` TFLOPS, timing CV `0.0008796200588309151`.
- Causal: max error `0.0078125`, median `433.8869934082031 ms`,
  `5.068193536474939` TFLOPS, timing CV `0.000364796510224527`.
- Geomean: `7.14948482274555` TFLOPS.
- Gate decision: accepted with reason `candidate established benchmark case set`.

Decision:

- Runtime source now carries `kMaxSeqLen=16384` and wrapper
  `SMOKE_SEQUENCES={16, 32, 64, 128, 256, 1024, 2048, 4096, 8192, 16384}`.
- Agent-side current MMA base sequences now include 16384, no-edit MMA scores below seq16384 are
  treated as below the accepted validation lane, and the exact seq16384 no-edit score is recorded.
- README and knowledge notes now describe seq16384 as the current accepted MMA lane.
- This establishes another target lane; it still does not satisfy the final objective of beating
  FlashAttention-2 on the target suite.

Verification:

- The loop itself compiled and scored the accepted candidate successfully on A6000/sm86.
- `uv run --extra dev pytest`: passed, 258 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.59: Bounded loop accepts seq32768 MMA lane

Success criteria for this checkpoint:

- Continue from the accepted seq16384 lane to the final configured long target shape.
- Preserve the accepted seq32768 source state and update validation constants, tests, README, and
  knowledge notes.
- Keep correctness and timing evidence attached to the accepted lane.

Live loop:

- Command:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 3 --loop-json attempts/loop_after_seq16384_lane.json --timeout-s 900 --env-file ../avo/.env.local`
- Step 1: planner emitted a structured seq32768 cap batch (`kMaxSeqLen=32768`, add `32768` to
  `SMOKE_SEQUENCES`) and compile-checked it. Compile passed with no spills, 40 registers, 1 barrier,
  and 9920 bytes shared memory.
- Step 2: planner reused the pending transform JSON and scored the seq32768 lane. Correctness
  passed and the suite-aware gate accepted a new benchmark lane.

Accepted seq32768 lane:

- Nested lineage commit: `195ea8c` (`evolve: accept candidate`).
- Workload: `seq_len=32768`, `total_tokens=32768`, `num_heads=16`, `head_dim=128`, BF16, both
  causal modes, `trials=3`, `warmup=1`, `repeats=1`.
- Noncausal: max error `0.000244140625`, median `873.764892578125 ms`,
  `10.066887668437108` TFLOPS, timing CV `0.0007010256101198647`.
- Causal: max error `0.0078125`, median `889.2239379882812 ms`,
  `4.94593805139101` TFLOPS, timing CV `0.006366806227785341`.
- Geomean: `7.056217313717174` TFLOPS.
- Gate decision: accepted with reason `candidate established benchmark case set`.

Decision:

- Runtime source now carries `kMaxSeqLen=32768` and wrapper
  `SMOKE_SEQUENCES={16, 32, 64, 128, 256, 1024, 2048, 4096, 8192, 16384, 32768}`.
- Agent-side current MMA base sequences now include 32768, no-edit MMA scores below seq32768 are
  treated as below the accepted validation lane, and the exact seq32768 no-edit score is recorded.
- README and knowledge notes now describe seq32768 as the current accepted MMA lane.
- This completes initial target-shape lane coverage. It still does not satisfy the final objective
  of beating FlashAttention-2 on the target suite.

Verification:

- The loop itself compiled and scored the accepted candidate successfully on A6000/sm86.
- `uv run --extra dev pytest`: passed, 258 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-08 - Checkpoint 4.60: Prevent uncontrolled FA2 optional-extra builds

Success criteria for this checkpoint:

- Keep the FA2 baseline installation path pinned to the Ampere build environment.
- Prevent `uv run --extra baseline ...` from building `flash-attn` before AVO can set
  `FLASH_ATTN_CUDA_ARCHS=80`, `MAX_JOBS=1`, and `NVCC_THREADS=1`.
- Document the baseline dependency boundary and add a regression test.

Observation:

- Running `uv run --extra cuda --extra baseline python -m avo env --env-file ../avo/.env.local`
  attempted to build `flash-attn==2.8.3` during dependency resolution, before AVO code ran. The
  build used the upstream broad default arch list (`sm_80`, `sm_90`, `sm_100`, `sm_120`) instead of
  the controlled A6000/Ampere build settings. The process was interrupted and left no installed
  `flash_attn` module.

Decision:

- Removed `flash-attn` from the `baseline` optional dependency group. FA2 must be installed
  explicitly with:
  `FLASH_ATTN_CUDA_ARCHS=80 MAX_JOBS=1 NVCC_THREADS=1 uv pip install flash-attn --no-build-isolation`.
- Updated the README baseline command to run `seed-baseline` from the existing `cuda` environment
  after the explicit install, rather than using `--extra baseline`.
- Added a test that parses `pyproject.toml` and asserts the `baseline` extra does not auto-install
  `flash-attn`.
- Updated the Ampere knowledge base with the dependency-resolution caveat.

Verification:

- `uv run --extra baseline python - <<'PY' ... importlib.util.find_spec('flash_attn') ... PY`:
  returned `None` without starting a FlashAttention build.
- `uv run --extra dev pytest tests/test_cli.py::test_baseline_extra_does_not_auto_install_flash_attn tests/test_cli.py::test_baseline_build_env_targets_flash_attn_ampere tests/test_cli.py::test_seed_baseline_rejects_missing_flash_attn_when_cuda_build_blocked -q`: passed, 3 tests.
- `uv run --extra dev pytest`: passed, 259 tests.
- `uv run --extra dev ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-09 - Checkpoint 4.61: Make CUDA preflights class-oriented

Success criteria for this checkpoint:

- Reduce prompt/context specificity around prior CUDA mistakes.
- Keep CUDA kernel evolution on the structured-transform interface.
- Make recurring failure-class promotion activate concrete structural preflight checks, not just
  supervisor advice.
- Reframe small-shape seed caps as safety fences while keeping the long-sequence target suite front
  and center.

Decision:

- Replaced the old planning-risk phrase tuple with class-level risk classifiers for predicted
  compile failures, predicted correctness failures, incomplete edits, and no-effect skeletons.
- Simplified repo context and retry feedback so it describes structural contracts rather than exact
  historical failures.
- Added an always-on MMA wrapper/kernel shape-contract preflight for shape-cap changes.
- Added promoted hard checks for recurring CUDA syntax delimiter failures and recurring WMMA
  fragment-shape failures, and persisted concrete `track_names` in `preflight_tracks.json`.
- Kept raw `.cu`/`.cuh` diffs rejected; materialized `candidate_transform` patches receive the
  structural checks before compile or score.

Verification:

- `.venv/bin/python -m pytest tests/test_agent.py tests/test_evolve.py`: passed, 204 tests.
- `.venv/bin/python -m pytest`: passed, 262 tests.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-09 - Checkpoint 4.62: Seed controlled FA2 baseline

Success criteria for this checkpoint:

- Finish the controlled FlashAttention-2 install without broad upstream arch builds.
- Seed the FA2 baseline on the realistic BF16 target suite rather than tiny smoke shapes.
- Close the install reproducibility gap exposed by CUDA 13 Python wheels that ship only
  `libcudart.so.13`.

Observation:

- The controlled source build with `FLASH_ATTN_CUDA_ARCHS=80`, `MAX_JOBS=3`, and
  `NVCC_THREADS=1` compiled all CUDA objects but initially failed at link time:
  `/usr/bin/ld: cannot find -lcudart`.
- The CUDA runtime library existed under the Python-installed CUDA root as
  `libcudart.so.13`, but the manual install environment did not include the link shim that AVO's
  extension environment helper can create.

Decision:

- Added `avo baseline-env`, which emits shell exports for the exact baseline build environment:
  CUDA root, `CUDACXX`, path variables, Ampere compile target, conservative parallelism limits, and
  the local `libcudart.so` link shim when needed.
- Updated the README and Ampere knowledge base so FA2 install uses:
  `eval "$(uv run --extra cuda python -m avo baseline-env)"` before
  `uv pip install flash-attn --no-build-isolation`.
- Installed `flash-attn==2.8.3` successfully after the CUDA runtime link path was made visible.

Seeded FA2 baseline:

- Nested lineage commit: `9bd118e` (`chore: seed baseline`).
- Command:
  `uv run --extra cuda python -m avo seed-baseline ./lineage --backend flash-attn --seq-lens 4096,8192,16384,32768 --total-tokens 32768 --num-heads 16 --head-dim 128 --dtype bf16 --causal both --trials 3 --repeats 1 --warmup 1 --timeout-s 900 --force`
- All 8 cases passed correctness.
- Geomean: `109.82622931666803` TFLOPS.
- Noncausal TFLOPS by sequence: 4096 `118.5607640200293`, 8192 `119.5489291820707`,
  16384 `119.95136120842285`, 32768 `121.57214180377122`.
- Causal TFLOPS by sequence: 4096 `101.4267666135992`, 8192 `104.82817098455511`,
  16384 `100.61461165062543`, 32768 `95.7262425503512`.

Verification:

- `.venv/bin/python -c "import flash_attn; print(getattr(flash_attn, '__version__', 'no_version'))"`:
  printed `2.8.3`.
- `uv run --extra cuda python -m avo env --env-file ../avo/.env.local`: reported
  `flash_attn_installed: true` and exact CUDA 13.0 Torch/NVCC compatibility.
- `uv run --extra cuda python -m avo baseline-env --format json`: emitted the selected CUDA 13 root
  and build environment.
- `.venv/bin/python -m pytest tests/test_cli.py -q`: passed, 24 tests.
- `.venv/bin/python -m pytest`: passed, 264 tests.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed in both runtime and paper repos.

## 2026-05-09 - Checkpoint 4.63: Repair structured transform planning loop

Success criteria for this checkpoint:

- Keep FlashAttention-2 as a comparison baseline without letting it block candidate lineage
  progress.
- Record the current MMA candidate on the full realistic target suite, even though it is far below
  FA2.
- Address the planner failure mode where the agent describes broad CUDA edits in prose without a
  representable `candidate_transform`.
- Keep the fix general: prefer tiny transform operations and class-level feedback over another
  phrase blacklist.

Research context:

- Exa search for structured code-edit interfaces found current evidence that raw unified diffs are
  brittle for LLM code editing because line offsets and fragmented hunks are hard for models to
  generate reliably. Relevant references:
  - "To Diff or Not to Diff? Structure-Aware and Adaptive Output Formats for Efficient LLM-based
    Code Editing": https://arxiv.org/html/2604.27296
  - "SWE-Edit: Rethinking Code Editing for Efficient SWE-Agent":
    https://arxiv.org/html/2604.26102
- The takeaway for this runtime is not to add larger raw CUDA diff guards. The interface should give
  the planner small, deterministic edit operations and make invalid prose-only edits recover into
  structured operations when that can be done unambiguously.

Baseline and lineage decision:

- Runtime commit `cfcc889` changed candidate lineage comparison so FA2 baseline payloads are
  recorded as baseline lanes, not candidate acceptance thresholds.
- `accepted_score_lanes()` now keeps baseline and candidate lanes separately for the same workload
  signature, so the summary can show the FA2 comparison lane and the candidate lane side by side.
- Regression tests cover baseline/candidate lane separation and candidate regression checks against
  prior candidates rather than FA2.

Current full-suite MMA candidate lane:

- Nested lineage commit: `0ca463a` (`evolve: accept mma full target lane`).
- Command:
  `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_mma_attention_seed.py --seq-lens 4096,8192,16384,32768 --total-tokens 32768 --num-heads 16 --head-dim 128 --dtype bf16 --causal both --trials 3 --repeats 1 --warmup 1 --timeout-s 900`
- All 8 cases passed correctness.
- Geomean: `7.283505507996481` TFLOPS.
- Noncausal TFLOPS: seq4096 `10.629656942098142`, seq8192 `10.38592794365928`,
  seq16384 `10.045924630314838`, seq32768 `10.057536029077653`.
- Causal TFLOPS: seq4096 `5.40697018480344`, seq8192 `5.252459141043947`,
  seq16384 `5.084148001620259`, seq32768 `4.917481283665701`.
- This establishes the full target-suite candidate lane, but it is still far below the FA2 baseline
  geomean `109.82622931666803` TFLOPS.

Planning-loop observations:

- `attempts/loop_after_full_target_lane.json` ran two bounded steps and failed before local
  compile/score:
  - Step 1 was rejected because conditional risk text such as "if compile fails ... undefined async
    copy symbols" was classified as an incomplete edit.
  - Step 2 described a broad cp.async CUDA change but omitted both `candidate_transform` and
    `candidate_patch`.
- After the first prompt/classifier repair, `attempts/loop_after_planning_repair.json` still failed
  both steps for prose-only CUDA edits. This showed that feedback alone was not enough.

Decision:

- Runtime commit `5ed61be` adds a tiny `add_include` transform operation, materialized by the
  orchestrator rather than emitted as a raw CUDA diff.
- The decision parser can infer a structured `add_include` transform from simple prose such as
  "Add cuda_pipeline_primitives.h include to MMA kernel" when the target source is unambiguous.
- Planning-risk classification now preserves decision-field boundaries before scanning risk
  windows, and conditional execution-risk language no longer trips the incomplete-edit classifier.
- Retry feedback and repo context now say that broad CUDA ideas must shrink to at most four exact
  transform steps, or become a no-edit diagnostic.
- `avo env` is now rejected for planner-interface/schema recovery. Planning failures must lead to a
  valid transform or a kernel-search diagnostic, not another environment stability check.

Live verification:

- `attempts/loop_after_add_include_transform.json` ran two bounded steps after the repair.
- Step 1 still chose a no-edit environment diagnostic; this exposed the planner-interface recovery
  gap, which `5ed61be` now guards in validation.
- Step 2 produced/inferred:
  `{"op":"add_include","path":"candidates/cuda_mma_attention/attention_kernel.cu","header":"cuda_pipeline_primitives.h"}`.
- The orchestrator materialized the transform, applied it, compiled the MMA kernel for sm_86, and
  cleaned up the compile-only patch.
- Compile evidence: ptxas reported 40 registers, 1 barrier, 9920 bytes shared memory, 0 spills.

Verification:

- `.venv/bin/python -m pytest tests/test_agent.py tests/test_evolve.py -q`: passed, 213 tests.
- `.venv/bin/python -m pytest`: passed, 276 tests.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed in the runtime repo.
- Runtime commit `5ed61be111f034d12d8f886245673db4f452bb32` was pushed and fetch-verified on
  `coder-2011/avo-ampere`.

## 2026-05-09 - Checkpoint 4.64: Require pending transform on follow-up scores

Success criteria for this checkpoint:

- Run one bounded post-repair loop under the new structured-transform interface.
- Capture any remaining loop-control gap exposed by the run.
- Prevent the agent from claiming it is scoring a compiled transform while actually scoring the
  unmodified candidate.

Live loop:

- Command:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 3 --loop-json attempts/loop_after_transform_planning_repair_commit.json --timeout-s 900 --env-file ../avo/.env.local`
- The loop did not produce an accepted candidate.
- Step 1 scored the full target candidate lane with `trials=1`, all cases correct, geomean
  `7.196038438735886` TFLOPS. The gate rejected it against the prior candidate geomean
  `7.283505507996481`.
- Step 2 failed planning validation because the planner tried another recorded environment
  stability diagnostic.
- Step 3 scored the full target lane again with `trials=3`, all cases correct, geomean
  `7.258262928778019` TFLOPS. The gate again rejected it against `7.283505507996481`.

Observation:

- The planner described these scores as follow-up scores for the compiled `add_include` transform,
  but the decisions omitted the pending `candidate_transform`.
- Because the compile-only patch had been cleaned up, omitting the transform meant the loop was just
  rescoring the unmodified MMA seed. This is exactly the kind of search-loop waste the prompt
  critique called out: it passes harness validation but does not advance kernel evolution.

Decision:

- Runtime commit `2b25427` updates attempt-history validation so a score following a successful
  compile-only transform must include the exact pending `candidate_transform` JSON from the
  follow-up signal.
- If the agent omits it, the step is rejected as `planning_missing_pending_transform` rather than
  running a misleading no-edit candidate score.
- A different transform family is still allowed, but it has to be represented as an actual
  `candidate_transform` or legacy non-CUDA patch.

Verification:

- `.venv/bin/python -m pytest tests/test_evolve.py -q`: passed, 59 tests.
- `.venv/bin/python -m pytest`: passed, 278 tests.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed in the runtime repo.
- Runtime commit `2b2542711631db5482f42f606971327b9e4cc55c` was pushed and fetch-verified on
  `coder-2011/avo-ampere`.

## 2026-05-09 - Checkpoint 4.65: Reframe transforms as semantic moves

Success criteria for this checkpoint:

- Correct the overly literal interpretation of "small tool calls" as tiny textual edits.
- Preserve deterministic primitive transform materialization while making the candidate unit a
  scoped, coherent semantic move.
- Prevent header-only/helper-only CUDA support edits from becoming standalone optimization
  candidates or pending follow-up scores.

Observation:

- The `add_include` transform from checkpoint 4.63 was useful as a primitive, but accepting it as a
  standalone candidate repeated the same underlying problem in a new form: the loop could make and
  validate a tiny textual change with no semantic value.
- The better rule is: make the smallest coherent transformation that preserves invariants and can
  be validated. "Small" means scoped, reviewable, recoverable, and tied to a clear hypothesis; it
  does not mean the smallest possible source hunk.

Decision:

- Runtime commit `0b70de7` keeps `add_include` as a primitive step, but marks it as support-only.
- Standalone support-only transforms are rejected. `add_include` may only appear in a semantic
  batch with a real dataflow or validation-contract change that uses it.
- The prompt, repo context, and retry feedback now talk about small coherent transformations and
  primitive steps as representation details, not as the optimization objective.
- Attempt history no longer treats a support-only compile as a pending transform that should be
  scored. Such attempts remain visible as compile-only diagnostics, but they do not drive a bogus
  follow-up score.
- Planning failures from support-only transforms are classified as
  `planning_support_only_transform` so the failure mode is visible without adding another phrase
  blacklist.

Verification:

- `.venv/bin/python -m pytest tests/test_agent.py tests/test_evolve.py -q`: passed, 219 tests.
- `.venv/bin/python -m pytest`: passed, 282 tests.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed in the runtime repo.
- Runtime commit `0b70de7db13a2dca53b8479a350edce6279c932f` was pushed and fetch-verified on
  `coder-2011/avo-ampere`.

## 2026-05-09 - Checkpoint 4.66: Infer exact semantic replacements

Success criteria for this checkpoint:

- Run the live loop after reframing transforms as semantic moves.
- Convert one concrete recurring failure into a better structured edit interface rather than a new
  guard.
- Keep placeholder or ambiguous prose rejected.

Live loop:

- Command:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 3 --loop-json attempts/loop_after_semantic_transform_contract.json --timeout-s 900 --env-file ../avo/.env.local`
- The loop produced no accepted candidate.
- Step 1 failed planning validation because the planner described a coherent K shared-memory
  staging change but omitted both structured edit channels.
- Step 2 failed on the pending-transform follow-up gate added in checkpoint 4.64.
- Step 3 was the useful signal: the planner proposed an exact semantic replacement in prose:
  replacing ``__shared__ __nv_bfloat16 probabilities[kScoreElements];`` with
  ``__shared__ __nv_bfloat16 probabilities[kScoreElements + 8];`` to test padding, but still omitted
  `candidate_transform`.

Decision:

- Runtime commit `70f04dd` teaches the decision parser to infer `replace_once` when the planner
  provides exact backtick-quoted source and replacement snippets and the target candidate file is
  clear.
- This is not a new CUDA phrase guard. It improves the interface: exact source-to-source semantic
  replacement prose can be recovered into a deterministic transform.
- Placeholder snippets containing `...` or `…` are not inferred, so vague prose still fails instead
  of becoming a malformed patch.

Verification:

- `.venv/bin/python -m pytest tests/test_agent.py tests/test_evolve.py -q`: passed, 221 tests.
- `.venv/bin/python -m pytest`: passed, 284 tests.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed in the runtime repo.
- Runtime commit `70f04ddf33941b6faa96c779238b9cb74569d1e3` was pushed and fetch-verified on
  `coder-2011/avo-ampere`.

## 2026-05-09 - Checkpoint 4.67: Post-replacement loop still needs semantic transform payloads

Success criteria for this checkpoint:

- Run a short live loop after adding exact replacement recovery.
- Verify whether the planner now reaches materialized semantic transforms.
- Record the remaining failure mode honestly if it does not.

Live loop:

- Command:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 2 --loop-json attempts/loop_after_exact_replacement_recovery.json --timeout-s 900 --env-file ../avo/.env.local`
- The loop produced no accepted candidate and executed no local compile or score command.
- Step 1 failed planning validation after the planner described a broad K shared-memory staging
  transformation but still omitted `candidate_transform`/`candidate_patch`.
- Step 2 failed planning validation after the planner proposed a header/include availability step
  for future async-copy dataflow. Under checkpoint 4.65 this is support-only unless batched with a
  real semantic dataflow or validation-contract change.

Decision:

- No runtime code was changed for this checkpoint. The exact-replacement recovery works for exact
  source-to-source prose, but this loop did not produce that shape.
- The remaining gap is broader than one phrase: the planner is still describing coherent CUDA
  transformations without serializing them into the structured transform channel.
- The next implementation work should improve semantic transform construction or planner feedback,
  not add another CUDA mistake phrase guard.

## 2026-05-09 - Checkpoint 4.68: Require scoped semantic edit modes

Success criteria for this checkpoint:

- Stop treating "small tool calls" as tiny text edits.
- Force the planner to choose an explicit edit channel: structured transform, legacy non-CUDA raw
  patch, or no-edit diagnostic.
- Verify that the loop can recover from a structural CUDA preflight failure into a materialized
  semantic transform and then score the exact pending transform.

Runtime change:

- Runtime commit `9248a6da70b011e738e9f01ad98471432991378f` adds required `edit_mode` values:
  `transform`, `legacy_patch`, and `no_edit`.
- Missing `edit_mode` is still inferred for backward-compatible stored attempts, but new strict tool
  calls must select the channel explicitly.
- `transform` mode now rejects prose-only CUDA changes and requires `candidate_transform`.
- Explicit `no_edit` mode now requires `candidate_edit` to start with `No edit;`.
- Prompt/schema language was changed from tiny textual operations to scoped semantic moves:
  primitive transform steps are materialization details, and the unit of search is the smallest
  coherent transformation that preserves invariants and can be validated.

Verification:

- `.venv/bin/python -m pytest tests/test_agent.py tests/test_evolve.py -q`: passed, 225 tests.
- `.venv/bin/python -m pytest`: passed, 288 tests.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed in the runtime repo.
- Runtime commit `9248a6da70b011e738e9f01ad98471432991378f` was pushed and fetch-verified on
  `coder-2011/avo-ampere`.

Live loop:

- Command:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 2 --loop-json attempts/loop_after_explicit_edit_mode.json --timeout-s 900 --env-file ../avo/.env.local`
- Step 1 proposed a coherent cp.async K-staging semantic batch, but structural preflight rejected it
  as `async_copy_granularity` because it used scalar BF16 async copies.
- Step 2 corrected the same transform family into 16-byte grouped async copies. The structured batch
  materialized, passed preflight, and compiled:
  `avo compile --source candidates/cuda_mma_attention/attention_kernel.cu --out-dir build/mma_k_async`.
- The compile used 40 registers and 18112 bytes shared memory. No candidate was accepted because the
  loop stopped at `max_steps` before scoring.

Follow-up score:

- Command:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 1 --loop-json attempts/loop_after_explicit_edit_mode_score.json --timeout-s 900 --env-file ../avo/.env.local`
- The next planner step obeyed the pending-transform contract and scored the exact compiled
  `candidate_transform` on the full target lane:
  `seq_lens 4096,8192,16384,32768`, `total_tokens 32768`, `num_heads 16`, `head_dim 128`, BF16,
  both causal modes.
- The candidate failed correctness in all 8 cases with `RuntimeError: candidate output contains
  non-finite values`, so geomean was `0.0` TFLOPS and the gate rejected it against the current best
  `7.283505507996481` TFLOPS.

Decision:

- This did not improve throughput, but it is the first loop in this sequence to turn a structural
  preflight class into a corrected structured CUDA transform, compile it, and force the follow-up
  score of the exact pending transform.
- The next search work should classify the non-finite cp.async K-staging failure as a transform
  family issue, likely around buffer swapping/wait ordering or incomplete current-buffer update,
  rather than adding another prompt phrase.

## 2026-05-09 - Checkpoint 4.69: Preflight async pipeline lifecycle failures

Success criteria for this checkpoint:

- Convert the non-finite cp.async K-staging score failure into a structural failure class.
- Keep the check class-oriented: reject invalid async pipeline lifecycles, not exact patch phrases.
- Verify the new check catches the real failed transform from checkpoint 4.68 while allowing a
  lifecycle-correct two-stage sketch.

Research basis:

- NVIDIA CUDA Programming Guide, "Asynchronous Data Copies": LDGSTS async copies are global-to-shared
  operations, the primitive API uses `__pipeline_memcpy_async`, `__pipeline_commit`, and
  `__pipeline_wait_prior`, and shared data requires `__syncthreads()` after the copy completion
  mechanism.
- NVIDIA CUDA Programming Guide, "Pipelines": `__pipeline_wait_prior(N)` / pipeline wait-prior waits
  for all but the last `N` commits. A two-stage prefetch pattern therefore must keep enough committed
  stages in flight before waiting with prior `1`.
- NVIDIA Ampere Tuning Guide: Ampere sm_86 supports hardware-accelerated global-to-shared async copy,
  but correctness still depends on valid synchronization and shared-memory lifecycle.

Runtime change:

- Runtime commit `8a6af5c33afbb6fddff158290d251dd8b9cd741f` adds structural preflight track
  `async_pipeline_stage_lifecycle`, classified as `correctness_nonfinite_output`.
- The detector rejects cp.async transforms that:
  - call `__pipeline_wait_prior(1)` before two committed stages have been inserted, or
  - use a `1 - stage`/`1 - buffer` double-buffer pattern without advancing that binary stage variable.
- Attempt-history classification now distinguishes score failures with non-finite output as
  `correctness_nonfinite_output` instead of generic `correctness_failed`, and includes the first score
  error in the attempt summary.

Verification:

- Focused tests:
  `.venv/bin/python -m pytest tests/test_agent.py::test_parse_variation_decision_rejects_invalid_async_pipeline_lifecycle tests/test_agent.py::test_structural_preflight_allows_valid_async_pipeline_lifecycle tests/test_evolve.py::test_summarize_attempt_history_classifies_nonfinite_score -q`: passed, 3 tests.
- `.venv/bin/python -m pytest tests/test_agent.py tests/test_evolve.py -q`: passed, 228 tests.
- `.venv/bin/python -m pytest -q`: passed, 291 tests.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed in the runtime repo.
- The actual failed score attempt from `attempts/2026-05-09T04-10-10-00-00.json` now fails
  `validate_candidate_patch_structural_preflight(..., allow_cuda_source_edits=True)` with:
  `structural preflight track async_pipeline_stage_lifecycle classified as correctness_nonfinite_output`.
- Runtime commit `8a6af5c33afbb6fddff158290d251dd8b9cd741f` was pushed and fetch-verified on
  `coder-2011/avo-ampere`.

Decision:

- This is a hard structural track derived from the failed transform family: a compiled async-copy
  transform that reads from a stage before it is guaranteed ready or never advances the stage index
  can produce non-finite outputs.
- The next loop should avoid this exact lifecycle class and either produce a valid two-stage
  K-staging transform or move to a different Ampere optimization family.

## 2026-05-09 - Checkpoint 4.70: Clear stale pending transform follow-ups

Success criteria for this checkpoint:

- Run a short loop after adding the async pipeline lifecycle preflight.
- If the planner fails before CUDA execution, classify whether the failure is stale loop state or a
  real kernel-search issue.
- Fix stale follow-up state without weakening the fresh compile-only score contract.

Live loop:

- Command:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 2 --loop-json attempts/loop_after_async_lifecycle_preflight.json --timeout-s 900 --env-file ../avo/.env.local`
- No compile or score command executed.
- Step 1 failed planning validation with:
  `next_command scores without the pending compile-only candidate_transform`.
- Step 2 failed planning validation with:
  `next_command repeats a recorded no-patch compile diagnostic`.

Diagnosis:

- The pending-transform gate was scanning for any historical compile-only transform that did not have
  the same transform identity in a score attempt.
- That made an older shape-cap compile from before the cp.async score remain pending even after a
  later score attempt had already moved the loop forward.
- The desired behavior is narrower: only the latest successful compile-only transform should be
  pending, and a later score attempt should clear older compile-only follow-ups. Planning failures
  after a fresh compile should still preserve the pending score requirement.

Runtime change:

- Runtime commit `60b831b80e0f2b35148c8985e5575bf72b85eb1b` changes
  `_pending_compile_only_transform` to scan backward until the first score payload or successful
  compile-only transform:
  - if the newest relevant event is a score, no stale compile follow-up remains pending;
  - if the newest relevant event is a compile-only transform, the exact-transform score requirement
    remains active.

Verification:

- Focused tests:
  `.venv/bin/python -m pytest tests/test_evolve.py::test_summarize_attempt_history_drops_stale_compile_followup_after_later_score tests/test_evolve.py::test_summarize_attempt_history_keeps_compile_followup_after_planning_failure tests/test_evolve.py::test_attempt_history_rejects_score_without_pending_transform -q`: passed, 3 tests.
- `.venv/bin/python -m pytest tests/test_agent.py tests/test_evolve.py -q`: passed, 229 tests.
- `.venv/bin/python -m pytest -q`: passed, 292 tests.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed in the runtime repo.
- Runtime commit `60b831b80e0f2b35148c8985e5575bf72b85eb1b` was pushed and fetch-verified on
  `coder-2011/avo-ampere`.

Decision:

- This was loop-state debt exposed by the post-preflight run, not a CUDA search result.
- The next loop should now be free to choose a new scored transform family without being blocked by
  an old compile-only transform that was superseded by a later score attempt.

## 2026-05-09 - Checkpoint 4.71: Accept smaller MMA block candidate

Success criteria for this checkpoint:

- Re-run the loop after clearing stale pending-transform state.
- Accept only a candidate that passes correctness on the full target lane and beats the current best
  geomean.
- Commit the accepted runtime source state and record lineage evidence.

Live loop:

- Command:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 2 --loop-json attempts/loop_after_stale_pending_fix.json --timeout-s 900 --env-file ../avo/.env.local`
- The loop stopped after one accepted candidate.
- Candidate transform:
  `set_constexpr_int` on `candidates/cuda_mma_attention/attention_kernel.cu`, changing
  `kThreads` from `256` to `128`.
- Hypothesis: the MMA kernel uses one warp for tensor-core math, so reducing the block from 256 to
  128 threads may reduce inactive thread overhead while preserving enough parallelism for
  cooperative loads/shared-memory initialization.

Score result:

- Command:
  `avo score --backend candidate --candidate candidates/cuda_mma_attention_seed.py --seq-lens 4096,8192,16384,32768 --total-tokens 32768 --num-heads 16 --head-dim 128 --dtype bf16 --causal both --repeats 1 --warmup 1 --timeout-s 300`
- Correctness: passed all 8 full-target BF16 cases.
- Geomean: `7.777584666360881` TFLOPS.
- Previous best: `7.283505507996481` TFLOPS.
- Gate decision: accepted, "candidate passed correctness and throughput gate".
- Per-case TFLOPS:
  - noncausal seq4096: `11.52054882841778`
  - causal seq4096: `5.459380958407761`
  - noncausal seq8192: `11.502745847543771`
  - causal seq8192: `5.3376604472323566`
  - noncausal seq16384: `11.273997007570372`
  - causal seq16384: `5.301768635035056`
  - noncausal seq32768: `11.123740467635363`
  - causal seq32768: `5.214821039250831`

Commits:

- Nested lineage commit: `cb558aa` (`evolve: accept candidate`), no remote configured for
  `/home/ubuntu/avo-ampere/lineage`.
- Runtime commit `3ab4a57a1ea679171950df239bb5cacf0b345154` applies the accepted `kThreads=128`
  source state and was pushed/fetch-verified on `coder-2011/avo-ampere`.

Verification:

- `.venv/bin/python -m pytest -q`: passed, 292 tests after applying the accepted source state.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed in the runtime repo.

Decision:

- This is a modest but real improvement, roughly `+6.78%` over the previous candidate geomean.
- It remains far below FA2 (`~109.8` TFLOPS recorded earlier), so the search should continue toward
  deeper Ampere dataflow and tiling changes. This checkpoint only establishes a better accepted
  baseline for subsequent agent iterations.

## 2026-05-09 - Checkpoint 4.72: Require semantic transform alignment

Success criteria for this checkpoint:

- Inspect the next loop after accepting the `kThreads=128` baseline.
- Fix the planner interface so "small" means the smallest coherent semantic move, not the smallest
  possible textual edit.
- Keep constant-only transforms valid when the constant is the real optimization, but reject them
  when the hypothesis claims a larger dataflow, tiling, staging, or scheduling behavior.

Loop result:

- Command:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 2 --loop-json attempts/loop_after_smaller_mma_block.json --timeout-s 900 --env-file ../avo/.env.local`
- The loop completed two steps and accepted no candidate.
- Step 1 compile-checked a `set_constexpr_int` transform changing `kThreads` from `128` to `256`,
  while the prose claimed it would support two concurrent 16-row query tiles per block. This was a
  semantic mismatch: the materialized edit only changed the block-size constant and did not implement
  the claimed query-tile dataflow.
- Step 2 scored a shared K/V staging transform on the full target lane. Correctness passed, but
  throughput regressed to geomean `4.040569721634079` TFLOPS versus best `7.777584666360881`, so the
  lineage gate rejected it.

Runtime change:

- Runtime commit `c4811a5b9aa1845a06ac02b523fd22b21e771ab5`
  (`fix: require semantic transform alignment`) adds a semantic-alignment preflight:
  - `set_constexpr_int` and `add_int_to_python_set` are treated as contract-only transforms;
  - contract-only transforms are allowed for real constant/shape-contract retunes;
  - they are rejected when planning text claims new executable dataflow, tiling, staging, or
    scheduling behavior that the transform does not materialize;
  - the transform batch cap was raised from 4 to 8 materialization steps so coordinated semantic
    edits are not forced into artificially tiny text changes.
- Attempt classification now records this as `planning_transform_semantic_mismatch`, promotable under
  the `semantic_transform_contract` track.

Verification:

- Focused tests:
  `.venv/bin/python -m pytest tests/test_agent.py::test_parse_variation_decision_allows_contract_only_constant_retune tests/test_agent.py::test_parse_variation_decision_rejects_constant_proxy_for_dataflow_claim tests/test_agent.py::test_decision_feedback_explains_transform_semantic_mismatch_error tests/test_evolve.py::test_summarize_attempt_history_classifies_transform_semantic_mismatch -q`: passed, 4 tests.
- `.venv/bin/python -m pytest tests/test_agent.py tests/test_evolve.py -q`: passed, 233 tests.
- `.venv/bin/python -m pytest -q`: passed, 296 tests.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed in the runtime repo.
- Runtime commit `c4811a5b9aa1845a06ac02b523fd22b21e771ab5` was pushed and fetch-verified on
  `coder-2011/avo-ampere`.

Decision:

- This does not add another CUDA optimization, but it addresses a root loop-interface failure exposed
  by the latest run.
- The next planner step should either make a constant-only hypothesis about the current invariant or
  materialize the full coherent dataflow change it claims, then validate it on the realistic target
  lane.

## 2026-05-09 - Checkpoint 4.73: Reject smaller MMA block retune

Success criteria for this checkpoint:

- Refresh the planner knowledge with Ampere FA2/CUTLASS cues before another loop.
- Run a bounded loop after the semantic-transform alignment fix.
- Treat a clean full-target rejection as search evidence, not a framework failure.

Research refresh:

- Exa found NVIDIA CUTLASS's Ampere FlashAttention v2 example, which uses 128x128 M/N tiles,
  128 threads, 16-byte contiguous alignment, Q/K/V shared-memory staging through `cp.async`,
  swizzled shared-memory layouts, Ampere tensor-core MMA, and integrated online softmax.
- Exa also surfaced FlashAttention SM80 kernel code where `NumThreads` is derived from the tiled
  MMA shape and used for launch/scheduling decisions. This reinforced the current interface rule:
  a thread-count constant retune is valid as a constant retune, but not as a proxy for unimplemented
  multi-query-tile or split-work dataflow.
- Runtime commit `abfdeb6fcb4280fac71c89ae84fe277020ffc6e1`
  (`docs: refresh ampere semantic search cues`) added this note to `knowledge/ampere.md` and was
  pushed/fetch-verified on `coder-2011/avo-ampere`.

Live loop:

- Command:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 2 --loop-json attempts/loop_after_semantic_alignment.json --timeout-s 900 --env-file ../avo/.env.local`
- Step 1 compiled a `set_constexpr_int` transform changing `kThreads` from `128` to `64`.
- Compile diagnostics: sm86 compile passed, 40 registers, 1 barrier, 9920 bytes shared memory,
  0 spill stores, and 0 spill loads.
- Step 2 scored the same `kThreads=64` transform on the full target lane with three timing trials.

Score result:

- Correctness: passed all 8 full-target BF16 cases.
- Geomean: `7.587127963961811` TFLOPS.
- Previous best: `7.777584666360881` TFLOPS.
- Gate decision: rejected, "candidate regressed geomean throughput".
- Per-case TFLOPS:
  - noncausal seq4096: `11.369773348943326`
  - causal seq4096: `5.397254654447017`
  - noncausal seq8192: `11.165628592873695`
  - causal seq8192: `5.2817504078010975`
  - noncausal seq16384: `11.040690799210282`
  - causal seq16384: `5.120668906329017`
  - noncausal seq32768: `10.764721410572763`
  - causal seq32768: `4.985489019182219`

Decision:

- The loop behaved correctly after the semantic-alignment fix: it made a coherent constant retune,
  compiled it, scored the exact transform, and rejected a throughput regression without leaving the
  worktree dirty.
- The result weakens the "fewer idle threads improves occupancy" hypothesis for this seed. The next
  search step should move away from block-size retuning and toward an actual Ampere dataflow change,
  preferably one that borrows a narrower piece of FA2/CUTLASS structure such as swizzled shared
  layout, Q/K/V copy granularity, or a real tiled-MMA work distribution.

## 2026-05-09 - Checkpoint 4.74: Add local searchable knowledge retrieval

Success criteria for this checkpoint:

- Give the Anthropic planner searchable local knowledge access instead of passing the entire
  `knowledge/ampere.md` file as one flat prompt block.
- Keep the implementation deterministic, auditable, and file-based before adding any larger external
  RAG service.
- Wire retrieval into `agent-plan`, `evolve-once`, and `evolve-loop` so retrieved context updates as
  lineage and attempt history change.

Preceding loop result:

- A loop after the rejected `kThreads=64` retune produced two useful signals:
  - planning step 1 failed with `planning_transform_semantic_mismatch` for another contract-only
    `kThreads=64` idea whose prose claimed broader register/shared-memory/dataflow effects;
  - planning step 2 moved to a real synchronous Q shared-memory staging transform.
- The Q-staging transform:
  - declared a static 16x128 BF16 Q shared-memory tile;
  - cooperatively loaded the current Q tile before the K loop;
  - synchronized;
  - changed QK WMMA loads to read tile-local shared memory instead of global Q.
- Correctness passed for all 8 full-target BF16 cases, but throughput regressed:
  - geomean `6.722112165053056` TFLOPS versus best `7.777584666360881`;
  - noncausal seq4096 `9.576218136406252`;
  - causal seq4096 `4.97938654742146`;
  - noncausal seq8192 `9.448375190110047`;
  - causal seq8192 `4.926014953814022`;
  - noncausal seq16384 `9.230892038439242`;
  - causal seq16384 `4.7701761514162415`;
  - noncausal seq32768 `9.217864326048506`;
  - causal seq32768 `4.6282297552780625`.
- Runtime commit `78cde3293db9562ae42594cb913bcc83e86b9f37`
  (`docs: record q staging regression`) added that negative evidence to `knowledge/ampere.md` and
  was pushed/fetch-verified.

Runtime change:

- Runtime commit `c21e4608a40a652e55274cce3ce9adb97313d257`
  (`feat: add local knowledge retrieval`) adds `avo/knowledge.py`.
- The retriever:
  - indexes supported local files (`.md`, `.txt`, `.py`, `.cu`, `.cuh`, `.cpp`, `.h`, `.hpp`);
  - chunks files into bounded line-range snippets;
  - scores chunks with deterministic lexical/idf-style matching;
  - returns an auditable prompt section with file labels, chunk ids, line ranges, and scores;
  - treats `knowledge/ampere.md` as the entry point for the surrounding `knowledge/` corpus.
- New CLI:
  `uv run python -m avo knowledge-search knowledge/ampere.md --query "Ampere cp.async shared staging WMMA head_dim 128"`
- `agent-plan`, `evolve-once`, and `evolve-loop` now build the retrieval query from:
  - the current lineage summary;
  - recent attempt history;
  - local repo context.

Verification:

- `uv run python -m avo knowledge-search knowledge/ampere.md --query "Ampere cp.async shared staging WMMA head_dim 128" --max-chunks 3 --max-chars 4000`: returned ranked local snippets with file/chunk metadata.
- `.venv/bin/python -m pytest tests/test_knowledge.py tests/test_cli.py::test_knowledge_search_command_prints_retrieved_context tests/test_cli.py::test_evolve_loop_runs_until_accepted_and_records_attempts -q`: passed, 5 tests.
- `.venv/bin/python -m pytest -q`: passed, 300 tests.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed in the runtime repo.
- Runtime commit `c21e4608a40a652e55274cce3ce9adb97313d257` was pushed and fetch-verified on
  `coder-2011/avo-ampere`.

Decision:

- This is the first version of the searchable knowledge base. It is intentionally local and
  deterministic rather than a full vector/RAG service.
- Next improvement, if needed, is to add more source documents under `knowledge/` (FA2 source notes,
  CUTLASS excerpts, profiler notes, and paper notes), because the retrieval substrate is now in place.

## 2026-05-09 - Checkpoint 4.75: Define and test retrieved knowledge claims

Success criteria for this checkpoint:

- Define the derived/found knowledge as explicit claims, not just loose notes.
- For each claim, state why it is useful to future search.
- Test retrieval against the real local corpus so we know the planner can actually recover the
  important facts.

Runtime change:

- Runtime commit `1c47fbcb42c0f3f2f6efb36daa68ded0e09ee60c`
  (`test: define and verify retrieved knowledge`) adds `knowledge/retrieval_claims.md`.
- The manifest defines high-value claims in four groups:
  - Ampere target and FA2 baseline constraints;
  - FA2/CUTLASS directional cues;
  - transform-interface lessons;
  - CUDA structural constraints and recent search evidence.
- Each claim records:
  - the claim itself;
  - evidence source;
  - why it is useful;
  - the retrieval query expected to recover it.

Claims now explicitly defined:

- A6000/sm86 means Ampere primitives (`cp.async`, `mma.sync`, warp reductions, BF16 tensor cores),
  not Blackwell TMA/WGMMA/FA4.
- FA2 is the comparison baseline, but not the lineage acceptance threshold.
- CUTLASS Ampere FA2 cues point toward 128x128 tiles, 128 threads, 16-byte alignment, Q/K/V staging
  through `cp.async`, swizzled shared-memory layouts, tensor-core MMA, and online softmax.
- SM80 FA code treats `NumThreads` as a tiled-MMA property, not as a proxy for unimplemented new
  workload distribution.
- `set_constexpr_int` and Python set transforms are contract-only; they must not claim new dataflow,
  tiling, staging, scheduling, split-Q work, or async pipelines.
- Recurring failures should become structural preflight tracks, not phrase bans.
- Future `cp.async` attempts need aligned 16-byte groups, 8 BF16 elements per 16-byte group, disjoint
  scalar tails, partial-tile guarding/zero-fill, and real overlap.
- WMMA BF16 edits should preserve the local seed's supported 16x16x16 contract unless the kernel is
  deliberately rewritten around a different TensorOp interface.
- Current best accepted candidate: `kThreads=128`, full-target correctness, geomean
  `7.777584666360881`.
- `kThreads=64` is correct but slower: geomean `7.587127963961811`.
- Isolated synchronous Q shared-memory staging is correct but slower: geomean
  `6.722112165053056`.

Verification:

- `tests/test_knowledge.py` now exercises the real `knowledge/ampere.md` entry point and its sibling
  corpus files.
- The tests verify retrieval of high-value facts for:
  - `cp.async` 16-byte/BF16/dataflow constraints;
  - CUTLASS Ampere FA2 tile/thread/softmax cues;
  - semantic transform mismatch and contract-only transforms;
  - synchronous Q-staging regression evidence;
  - FA2 baseline versus lineage threshold;
  - WMMA 16x16x16 shape contract;
  - rejected `kThreads=64` evidence.
- A manifest-wide test parses every `Retrieval query:` from `knowledge/retrieval_claims.md` and
  requires each query to retrieve the manifest context with `Why useful`.
- A manual audit over all 11 manifest queries showed relevant top-3 chunks, for example:
  - `Ampere sm86 A6000 ...` -> `retrieval_claims.md#0`, `ampere.md#0`;
  - `Ampere cp.async 16-byte ...` -> `ampere.md#3`, `ampere.md#4`,
    `retrieval_claims.md#1`;
  - `synchronous Q shared memory staging regression ...` -> `retrieval_claims.md#2`,
    `ampere.md#34`, `ampere.md#19`.
- `.venv/bin/python -m pytest tests/test_knowledge.py -q`: passed, 14 tests.
- `.venv/bin/python -m pytest -q`: passed, 311 tests.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed in the runtime repo.
- Runtime commit `1c47fbcb42c0f3f2f6efb36daa68ded0e09ee60c` was pushed and fetch-verified on
  `coder-2011/avo-ampere`.

Decision:

- The KB now contains explicit, useful, retrievable claims rather than only chronological notes.
- The retrieval layer still needs more source depth, but the current corpus is test-covered enough to
  use safely in the next planner loop.

## 2026-05-09 - Checkpoint 4.76: Add general CUDA grounding to the KB

Success criteria for this checkpoint:

- Move beyond Ampere/search-specific notes and give the planner general CUDA working knowledge.
- Keep the new material practical: how CUDA works, how CUDA programmers reason, and what mistakes
  to avoid.
- Verify that the information is retrievable through the existing `knowledge/ampere.md` entry point.

Runtime change:

- Runtime commit `1240062233177e189a0c2e4d7ef1145a315ac452`
  (`docs: add general cuda knowledge grounding`) adds `knowledge/b/cuda_general.md`.
- The new KB file covers:
  - execution hierarchy: grid, block, thread, warp, SIMT, divergence, and indexing;
  - memory spaces: global, shared, registers, local memory, constant memory, and cache/shared-memory
    tradeoffs;
  - global-memory coalescing, vectorized/aligned access, and redundant read avoidance;
  - shared-memory staging, synchronization, bank conflicts, and tiling;
  - occupancy/resource tradeoffs involving block size, registers, shared memory, spills, and ptxas
    output;
  - profiling and measurement with correctness first, warmups, repeated samples, medians, and
    bottleneck classification;
  - the practical CUDA optimization workflow: hypothesis, measurement, one coherent transform,
    invariant preservation, and treating regressions as evidence.
- `knowledge/retrieval_claims.md` now has a `General CUDA Grounding` section with retrievable claims
  for execution model, memory spaces, coalescing, shared memory, occupancy, and optimization
  workflow.
- `knowledge/ampere.md` points planners to `knowledge/b/cuda_general.md` for broad CUDA grounding
  while keeping itself focused on Ampere/attention/local search evidence.
- README now documents the split between general CUDA grounding under `knowledge/b/` and
  Ampere-specific evidence in `knowledge/ampere.md`.

Sources:

- NVIDIA CUDA Programming Guide for SIMT kernels, thread hierarchy, memory spaces, synchronization,
  coalescing, and shared memory.
- NVIDIA CUDA C++ Programming Guide for hardware multithreading, resource residency, registers,
  shared memory, and occupancy.
- NVIDIA CUDA C++ Best Practices Guide for iterative optimization, timing, shared-memory tiling, and
  performance measurement.
- NVIDIA Nsight Compute Kernel Profiling Guide for profiler workflow and metric collection strategy.

Verification:

- `.venv/bin/python -m pytest tests/test_knowledge.py -q`: passed, 21 tests.
- Manual retrieval:
  `knowledge-search knowledge/ampere.md --query "CUDA execution model grid block thread warp SIMT divergence memory spaces occupancy profiling workflow" --max-chunks 6 --max-chars 9000`
  returned `retrieval_claims.md` plus `b/cuda_general.md` chunks covering the execution model,
  memory spaces, occupancy, profiling, and optimization workflow.
- `.venv/bin/python -m pytest -q`: passed, 318 tests.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed in the runtime repo.
- Runtime commit `1240062233177e189a0c2e4d7ef1145a315ac452` was pushed and fetch-verified on
  `coder-2011/avo-ampere`.

Decision:

- The KB is now less overfit to local CUDA failure history. It has a reusable CUDA mental model that
  should help the planner form better first hypotheses before relying on Ampere-specific details.

## 2026-05-09 - Checkpoint 4.77: Reject no-effect shared staging transforms

Success criteria for this checkpoint:

- Use the expanded CUDA KB in a real bounded planner loop.
- Treat the result as search-loop evidence, even if no candidate is accepted.
- Convert the observed failure into a structural preflight, not another phrase ban.

Research/context refresh:

- Exa over NVIDIA CUDA/Nsight docs reconfirmed that CUDA optimization should be
  measurement/profiling-led: coalescing, shared-memory reuse, occupancy/resource limits, and
  profiler bottleneck classification should drive edits.
- Exa over CUTLASS Ampere FlashAttention v2 reconfirmed the useful attention direction: tiled Q/K/V
  movement through shared memory with `cp.async`, tensor-core MMA, register pipelining, and online
  softmax.

Loop:

- Command:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 3 --loop-json attempts/loop_after_general_cuda_grounding.json --timeout-s 900 --env-file ../avo/.env.local`
- Result: no accepted candidate; `completed_steps=3`, `stopped_reason=max_steps`.
- Step 1 failed planning validation. The planner described cooperative double-buffered async
  K-staging in prose but did not serialize the transform as `candidate_transform`.
- Step 2 compiled a materialized transform that only added `cooperative_groups.h` and declared
  `__shared__ __nv_bfloat16 k_shared[2][kTile * kHeadDim]`.
- Compile passed, but ptxas warned that `k_shared` was declared and never referenced. No correctness
  or TFLOPS score was run. This was not a meaningful semantic move.
- Step 3 failed planning validation again by trying to score the same no-effect transform in prose.

Runtime change:

- Runtime commit `2e3efd0c73cdbbca25afe21ff427c2d74c379457`
  (`fix: reject unused shared staging transforms`) adds a structural preflight track:
  `no_effect_shared_staging_buffer`, classified as `no_effect_or_skeleton`.
- The detector rejects a new CUDA `__shared__` staging buffer when the same materialized patch does
  not also load from, store to, or consume that buffer in executable dataflow.
- This is intentionally structural: it does not ban shared-memory staging, only a declaration-only
  staging scaffold.
- `knowledge/ampere.md` and `knowledge/retrieval_claims.md` now record the loop result and the new
  shared-staging-buffer claim.

Verification:

- Focused tests:
  `.venv/bin/python -m pytest tests/test_agent.py::test_structural_preflight_rejects_unused_shared_staging_buffer tests/test_agent.py::test_structural_preflight_allows_used_shared_staging_buffer tests/test_evolve.py::test_run_decision_command_rejects_materialized_unused_shared_buffer tests/test_knowledge.py -q`
  passed, 25 tests.
- `.venv/bin/python -m pytest -q`: passed, 322 tests.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed in the runtime repo.
- Runtime commit `2e3efd0c73cdbbca25afe21ff427c2d74c379457` was pushed and fetch-verified on
  `coder-2011/avo-ampere`.

Decision:

- The general CUDA KB moved the planner toward the right optimization family, but the transform
  interface still allowed under-materialized CUDA scaffolding. The new preflight closes that gap
  generally: future shared-memory staging must include producer/consumer dataflow in the same
  coherent transform.

## 2026-05-09 - Checkpoint 4.78: Clear stale compile-only followups after preflight rejection

Success criteria for this checkpoint:

- Run the loop again with `no_effect_shared_staging_buffer` active.
- Verify that declaration-only shared-memory staging is blocked before scoring.
- Fix any stale attempt-history signal created by that rejection.

Loop:

- Command:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 3 --loop-json attempts/loop_after_shared_staging_preflight.json --timeout-s 900 --env-file ../avo/.env.local`
- Result: no accepted candidate; `completed_steps=3`, `stopped_reason=max_steps`.
- Step 1 failed planning validation because it again tried to score the previous compile-only
  shared-staging transform in prose.
- Step 2 supplied the exact transform and attempted a full-target score, but runtime preflight
  rejected it before scoring:
  `structural preflight track no_effect_shared_staging_buffer classified as no_effect_or_skeleton`.
- Step 3 again failed planning validation by asking to score the already-rejected no-effect
  transform.

Runtime change:

- Runtime commit `cfcaa1cb5521647c03426cb47b2d741a00afc540`
  (`fix: clear rejected transform followups`) updates attempt-history follow-up logic.
- `_pending_compile_only_transform` now tracks later structural-preflight rejections by
  `candidate_transform` identity while scanning backward through attempts.
- If the same transform was structurally rejected after a compile-only success, the old
  compile-only success no longer produces a "score this candidate_transform" follow-up.
- `_has_successful_compile_only_transform` also ignores identities that have since been rejected by
  structural preflight.
- Knowledge notes and retrieval claims now record that structural rejection invalidates stale
  compile-only score follow-ups for the same transform identity.

Verification:

- Actual recent attempt summary after the fix no longer contains the stale
  `compiled successfully but has not been scored` follow-up. It instead reports recurring
  `planning_transform_preflight` and the active promoted transform-materialization track.
- Focused regression:
  `.venv/bin/python -m pytest tests/test_evolve.py::test_summarize_attempt_history_drops_compile_followup_after_preflight_rejection tests/test_evolve.py::test_summarize_attempt_history_requests_score_after_compile_only_transform tests/test_evolve.py::test_summarize_attempt_history_keeps_compile_followup_after_planning_failure tests/test_evolve.py::test_summarize_attempt_history_drops_stale_compile_followup_after_later_score -q`
  passed, 4 tests.
- Knowledge/follow-up focused run:
  `.venv/bin/python -m pytest tests/test_evolve.py::test_summarize_attempt_history_drops_compile_followup_after_preflight_rejection tests/test_knowledge.py -q`
  passed, 24 tests.
- `.venv/bin/python -m pytest -q`: passed, 324 tests.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed in the runtime repo.
- Runtime commit `cfcaa1cb5521647c03426cb47b2d741a00afc540` was pushed and fetch-verified on
  `coder-2011/avo-ampere`.

Decision:

- The loop is now less likely to get trapped by its own stale compile-only memory. A preflight
  rejection of a transform is treated as stronger evidence than an older compile success for the same
  transform.

## 2026-05-09 - Checkpoint 4.79: Score chunk unroll and improve ambiguous-anchor feedback

Success criteria for this checkpoint:

- Run a new loop after stale rejected-transform follow-ups were cleared.
- Score any coherent compile-only transform that the loop produces.
- If the transform interface blocks semantic CUDA edits, improve the repair signal without opening
  raw CUDA diffs.

Loop 1:

- Command:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 3 --loop-json attempts/loop_after_rejected_followup_clear.json --timeout-s 900 --env-file ../avo/.env.local`
- Result: no accepted candidate; `completed_steps=3`, `stopped_reason=max_steps`.
- Step 1 proposed a genuine K-staging transform: add a 16x128 BF16 `k_shared` tile, cooperatively
  stage K, and route the first QK WMMA fragment through shared memory. It failed materialization
  because `insert_before_once` found 3 matching anchors.
- Step 2 proposed a narrower K-staging transform and again failed materialization because the anchor
  was still ambiguous, now with 2 matches.
- Step 3 reset to a simpler WMMA chunk-loop unroll-by-2 transform for QK and PV loops. It compiled
  cleanly: no spills, 39 registers, 1 barrier, 9920 bytes shared memory. Because the loop hit
  max-steps, the rejected patch was cleaned up and the transform was left as a pending compile-only
  follow-up.

Loop 2:

- Command:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 2 --loop-json attempts/loop_after_chunk_unroll2_compile.json --timeout-s 900 --env-file ../avo/.env.local`
- Step 1 scored the exact pending chunk-unroll transform on the full target suite.
- Correctness: passed all 8 BF16 full-target cases.
- Geomean: `7.758592599549404` TFLOPS.
- Previous best: `7.777584666360881` TFLOPS.
- Gate: rejected for geomean regression.
- Per-case TFLOPS:
  - noncausal seq4096: `11.422010794038217`;
  - causal seq4096: `5.520844568626744`;
  - noncausal seq8192: `11.650990264230991`;
  - causal seq8192: `5.376679254549787`;
  - noncausal seq16384: `11.147879962119449`;
  - causal seq16384: `5.269909715075147`;
  - noncausal seq32768: `10.902247712441563`;
  - causal seq32768: `5.189518007452957`.
- Step 2 then tried to return to K staging but again failed planning validation by describing the
  edit in prose without a serialized transform.

Runtime change:

- Runtime commit `d87b630215bdb83c9192ecfd1a227347fafa1567`
  (`fix: explain ambiguous transform anchors`) improves materialization errors.
- Ambiguous `replace_once` and `insert_*_once` now report matching start line numbers and explicitly
  tell the planner to use a larger unique anchor with surrounding code.
- This keeps the structured-transform interface small while making semantic CUDA transform repair
  easier.
- Knowledge notes and retrieval claims now record:
  - K-staging failed on ambiguous anchors, not on CUDA structural invalidity;
  - chunk-loop unroll-by-2 was correct but regressed;
  - future work should prefer dataflow/layout/staging changes over more local unroll-only edits.

Verification:

- Focused transform tests:
  `.venv/bin/python -m pytest tests/test_evolve.py::test_materialize_candidate_transform_rejects_ambiguous_anchor tests/test_evolve.py::test_materialize_candidate_transform_reports_ambiguous_insert_lines -q`
  passed, 2 tests.
- Focused knowledge/transform run:
  `.venv/bin/python -m pytest tests/test_evolve.py::test_materialize_candidate_transform_rejects_ambiguous_anchor tests/test_evolve.py::test_materialize_candidate_transform_reports_ambiguous_insert_lines tests/test_knowledge.py -q`
  passed, 27 tests.
- `.venv/bin/python -m pytest -q`: passed, 327 tests.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed in the runtime repo.
- Runtime commit `d87b630215bdb83c9192ecfd1a227347fafa1567` was pushed and fetch-verified on
  `coder-2011/avo-ampere`.

Decision:

- The loop is now making more semantically useful CUDA proposals. The immediate transform-interface
  blocker is ambiguous anchors, so the next loop should be able to repair K-staging attempts with
  concrete line-number feedback instead of falling back to broad raw diffs.

## 2026-05-09 - Checkpoint 4.80: Expand general CUDA practice knowledge

Success criteria for this checkpoint:

- Move the knowledge base away from narrow CUDA failure anecdotes and toward general CUDA working
  knowledge.
- Keep the new information under `knowledge/b/`, where the broad background corpus lives.
- Make the information useful to the planner by adding retrieval claims and tests, not just prose.

Runtime change:

- Runtime commit `99f142c807b0dc0a28c7f5831e08a2079144e368`
  (`docs: expand general cuda practice knowledge`) adds
  `knowledge/b/cuda_programming_practice.md`.
- The new file covers:
  - CUDA as a throughput-oriented host/device programming model;
  - decomposing work across threads, warps, blocks, and tiles;
  - indexing, `threadIdx.x` linearization, layout, coalescing, vectorized access, and tail guards;
  - global/shared/register/local/constant memory tradeoffs;
  - shared-memory tiling as complete dataflow, including load mapping, layout conversion, barriers,
    bank conflicts, reuse, and overwrite safety;
  - synchronization scope: `__syncthreads()`, warp shuffles, atomics, and lack of a normal
    cross-block barrier;
  - double-buffering and `cp.async`/`cuda::memcpy_async` as producer/consumer overlap rather than
    cosmetic copy replacement;
  - tensor-core/WMMA contract thinking: fragment opacity, shape/layout/type/leading-dimension
    coupling, loads, MMA operations, and stores;
  - launch-resource tradeoffs: occupancy, registers, shared memory, block size, ptxas output, and
    launch feasibility;
  - streams, events, default-stream synchronization, and host/device timing discipline;
  - realistic-workload profiling with Nsight-style launch, occupancy, SpeedOfLight, scheduler,
    memory-workload, source-counter, and timing-distribution checks;
  - the desired planner rule: make the smallest coherent semantic transformation that preserves
    invariants and can be validated.
- `knowledge/retrieval_claims.md` now defines useful claims for those general topics, including an
  evidence source, why each claim matters, and the query expected to retrieve it.
- `tests/test_knowledge.py` now checks retrieval for the new general CUDA topics and verifies that
  the broad `knowledge/b` content is indexed from the `knowledge/ampere.md` entrypoint.
- `README.md` documents the new `knowledge/b/cuda_programming_practice.md` file.

Source provenance:

- Official NVIDIA CUDA Programming Guide pages used for the general model, SIMT kernels,
  memory spaces, streams/events, pipelines, and async-copy behavior:
  - https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html
  - https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html
  - https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html
  - https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html
  - https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html
- Official NVIDIA CUDA C++ Best Practices Guide used for coalescing, shared memory, bank conflicts,
  realistic workload guidance, and profiling workflow:
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- Official NVIDIA Nsight Compute Profiling Guide used for broad bottleneck categories such as
  launch metrics, occupancy, SpeedOfLight, scheduler behavior, memory workload, source counters,
  replay overhead, and roofline analysis:
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- Local source of derived guidance:
  - the user's design correction that "small" means scoped, reviewable, recoverable, hypothesis-led
    semantic transformations, not the smallest possible text edit;
  - recent AVO failures around no-effect shared-memory scaffolding, malformed async-copy attempts,
    WMMA fragment-shape edits, and compile-only transforms that did not improve the target score.

Verification:

- `.venv/bin/python -m pytest tests/test_knowledge.py -q`: passed, 34 tests.
- `.venv/bin/python -m pytest -q`: passed, 336 tests.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed in the runtime repo.
- Runtime commit `99f142c807b0dc0a28c7f5831e08a2079144e368` was pushed and fetch-verified on
  `coder-2011/avo-ampere`.

Decision:

- The knowledge base now has broader CUDA background that should help the planner form better
  hypotheses before editing kernels. This is a better root-level improvement than adding more
  reactive phrase bans around individual failed attempts.

## 2026-05-09 - Checkpoint 4.81: Ensure agent receives CUDA practice context

Success criteria for this checkpoint:

- Fix the agent-side path, not only the documentation path.
- Verify that the live planning context retrieves the broad CUDA practice content from
  `knowledge/b/cuda_programming_practice.md`.
- Keep the change minimal and scoped to planner-context construction and retrieval tests.

Issue:

- After checkpoint 4.80, `knowledge/b/cuda_programming_practice.md` existed and its targeted
  retrieval tests passed, but the actual agent planning context still did not include it under the
  current lineage/attempt/repo query.
- Direct check before the fix:
  - command constructed `_planning_context(...)` for the live runtime repo;
  - result: `"b/cuda_programming_practice.md" in knowledge` was `False`;
  - the top retrieved chunks were dominated by high-frequency local lineage and attempt-history terms.

Runtime change:

- Runtime commit `b397fd400a278afc0956a0adc0ee41a480c874ea`
  (`fix: include cuda practice in agent context`) adds a stable broad-CUDA practice retrieval anchor
  in `avo/cli.py`.
- `_planning_context` now builds the normal dynamic context first, then appends a bounded
  supplemental broad CUDA practice context only if the normal retrieval did not already include
  `b/cuda_programming_practice.md`.
- The supplemental retrieval uses section-heading terms, so it retrieves the actual practice content
  such as `# CUDA Kernel Design Practice`, `The Basic Mental Model`, `Decomposing Work`, and
  `Semantic Transform Guidance For The Planner`, rather than only the manifest/query-list chunk.
- This keeps the agent grounded in general CUDA practice while preserving the existing local
  lineage/attempt-history retrieval.

Verification:

- Live planning-context check after the fix:
  - `"b/cuda_programming_practice.md" in knowledge`: `True`;
  - `"CUDA Kernel Design Practice" in knowledge`: `True`;
  - supplemental broad CUDA practice context starts after the normal dynamic context and contains the
    actual guidance that a CUDA design starts from thread/warp/block work ownership.
- Focused tests:
  `.venv/bin/python -m pytest tests/test_cli.py::test_planning_context_includes_general_cuda_practice tests/test_knowledge.py::test_agent_planning_query_retrieves_cuda_practice_context -q`
  passed, 2 tests.
- `.venv/bin/python -m pytest -q`: passed, 338 tests.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed in the runtime repo.
- Runtime commit `b397fd400a278afc0956a0adc0ee41a480c874ea` was pushed and fetch-verified on
  `coder-2011/avo-ampere`.

Source/provenance note:

- Exa search during this checkpoint again surfaced the same official NVIDIA references as the right
  source class for this fix: CUDA Programming Guide SIMT kernels, CUDA Best Practices Guide, Nsight
  Compute guide, and Ampere tuning guide. No new external implementation pattern was copied.

Decision:

- This closes the specific gap where useful general CUDA knowledge existed locally but was not
  reaching the agent under realistic planning queries. The next useful check is to run another bounded
  evolve loop and see whether the planner uses this context to produce a coherent CUDA transform
  instead of falling back to prose or narrow historical fixes.

## 2026-05-09 - Checkpoint 4.82: Reject unsupported score-as-profile plans

Success criteria for this checkpoint:

- Run a bounded loop after the CUDA-practice context fix.
- Classify what the agent actually did with the fixed context.
- Fix any controllable agent-interface issue exposed by that loop.

Loop:

- Command:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 3 --loop-json attempts/loop_after_cuda_practice_context.json --timeout-s 900 --env-file ../avo/.env.local`
- Result: no accepted candidate; `completed_steps=3`, `stopped_reason=max_steps`.
- Step 1:
  - The agent used the broader CUDA-practice framing and correctly asked to classify the bottleneck
    before adding async/staging complexity.
  - The invalid part: it claimed it would profile memory bandwidth, tensor-core utilization,
    occupancy, and instruction mix, but the bounded command was only `avo score`.
  - `avo score` can measure correctness, timing, and TFLOPS; it does not run Nsight or collect
    profiler metrics.
  - The score passed correctness but was a no-edit full-target rerun and regressed versus the best:
    geomean `7.74684311206423` TFLOPS versus best `7.777584666360881`.
- Step 2:
  - Planning failed after three retries because the agent repeated a recorded no-patch compile
    diagnostic instead of attaching a `candidate_transform` or scoring a pending compiled transform.
- Step 3:
  - Planning failed after three retries because the agent described a Q shared-memory staging edit in
    prose but did not serialize the required `candidate_transform`.

Runtime change:

- Runtime commit `412e751cab081629b094aaabfa362441d87bc0e0`
  (`fix: reject unsupported score profiling plans`) tightens the agent validation.
- `avo score` plans are now rejected if the planning text asks for profiler-only evidence such as
  memory bandwidth, occupancy, scheduler behavior, instruction mix, roofline, stalls, bank conflicts,
  or tensor-core utilization.
- The validation error now explains that `avo score` reports correctness, timing, and TFLOPS only.
- The recorded no-patch compile feedback now tells the planner to score the exact pending compiled
  transform or include a source-changing `candidate_transform`, not to score/compile the unmodified
  seed again.
- The full target-suite no-edit MMA score
  `seq_lens=4096,8192,16384,32768 / total_tokens=32768 / num_heads=16 / head_dim=128`
  is now treated as a recorded unpatched seed score, so the loop should not spend another step
  remeasuring the same accepted candidate without a transform.

Verification:

- Focused tests:
  `.venv/bin/python -m pytest tests/test_agent.py::test_parse_variation_decision_rejects_score_claiming_profiler_metrics tests/test_agent.py::test_parse_variation_decision_rejects_recorded_unpatched_mma_target_suite_score tests/test_agent.py::test_decision_feedback_explains_recorded_no_patch_compile_error tests/test_agent.py::test_decision_feedback_explains_score_profiler_metric_error -q`
  passed, 4 tests.
- Related evolve classification tests:
  `.venv/bin/python -m pytest tests/test_evolve.py::test_summarize_attempt_history_classifies_planning_validation_failure tests/test_evolve.py::test_summarize_attempt_history_does_not_fingerprint_different_planning_errors -q`
  passed, 2 tests.
- `.venv/bin/python -m pytest -q`: passed, 341 tests.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed in the runtime repo.
- Runtime commit `412e751cab081629b094aaabfa362441d87bc0e0` was pushed and fetch-verified on
  `coder-2011/avo-ampere`.

Decision:

- The broad CUDA context changed the agent's reasoning in the right direction, but the action channel
  was still too permissive: it allowed a score command to masquerade as profiling. The fixed validator
  keeps the loop honest about what evidence each command can actually produce and should push the next
  loop toward either a real `candidate_transform` or a genuinely supported diagnostic.

## 2026-05-09 - Checkpoint 4.83: Reject env-as-source-inspection plans

Success criteria for this checkpoint:

- Run a short follow-up loop after the score/profiling guard.
- Fix the next controllable agent-command contract violation if one appears.
- Verify the fix with focused and full tests.

Loop:

- Command:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 2 --loop-json attempts/loop_after_score_profile_guard.json --timeout-s 900 --env-file ../avo/.env.local`
- Result: no accepted candidate; `completed_steps=2`, `stopped_reason=max_steps`.
- Step 1:
  - The agent no longer used `avo score` as a fake profiler.
  - It instead proposed "file inspection only" of warp-row and MMA candidate sources, but selected
    `avo env` as the bounded command.
  - `avo env` succeeded, but it only reports CUDA/build environment state; it cannot inspect source
    files. This was another action-channel mismatch.
- Step 2:
  - Planning failed after three retries on the already-known recorded no-patch compile diagnostic.

Runtime change:

- Runtime commit `d590888ec1bf64b51c7905da7290d08fc42788e6`
  (`fix: reject env source inspection plans`) adds an explicit env-command validator.
- `avo env` is now rejected when the planning text asks to inspect/read/review/examine source,
  files, kernels, wrappers, candidates, seeds, staging, or vectorization.
- The feedback says repo context already includes local candidate excerpts, so the planner should
  choose a structured `candidate_transform`, a compile/score diagnostic tied to a concrete source
  change, or a real CUDA/build environment diagnostic.
- Concrete CUDA/build failures still allow `avo env`; the guard targets source-inspection misuse,
  not legitimate environment diagnostics.

Verification:

- Focused tests:
  `.venv/bin/python -m pytest tests/test_agent.py::test_parse_variation_decision_rejects_env_for_source_inspection tests/test_agent.py::test_parse_variation_decision_allows_env_for_cuda_environment_check tests/test_agent.py::test_parse_variation_decision_allows_env_for_concrete_build_failure tests/test_agent.py::test_decision_feedback_explains_env_source_inspection_error -q`
  passed, 4 tests.
- Live failure-shape reproduction now raises:
  `next_command avo env cannot inspect source files; repo context already includes candidate excerpts, and env is only for CUDA/build environment diagnostics`.
- `.venv/bin/python -m pytest -q`: passed, 342 tests.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed in the runtime repo.
- Runtime commit `d590888ec1bf64b51c7905da7290d08fc42788e6` was pushed and fetch-verified on
  `coder-2011/avo-ampere`.

Decision:

- The current agent loop is now stricter about matching a bounded command to the evidence it can
  actually produce: `score` cannot pretend to be profiling, and `env` cannot pretend to inspect
  source. This is still not a performance breakthrough, but it is a root-level agent fix rather than
  another CUDA phrase ban.

## 2026-05-09 - Checkpoint 4.84: Surface transform anchor repair signal

Success criteria for this checkpoint:

- Run a follow-up loop after the env/source-inspection guard.
- Verify whether the agent now reaches a real structured CUDA transform.
- If transform materialization fails, improve the repair signal rather than adding another narrow
  phrase ban.

Loop:

- Command:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 2 --loop-json attempts/loop_after_env_source_guard.json --timeout-s 900 --env-file ../avo/.env.local`
- Result: no accepted candidate; `completed_steps=2`, `stopped_reason=max_steps`.
- Step 1:
  - The agent produced a real semantic CUDA transform: stage the full 16x128 Q tile cooperatively
    into shared memory once per CTA, then load WMMA Q fragments from `q_shared` instead of global Q.
  - This is directionally useful: it is a coherent Q-staging move tied to reuse/global-memory
    traffic, not a raw CUDA diff, not a fake profiler call, and not an env/source-inspection step.
  - Materialization failed before compile because the transform used `anchor: "  __syncthreads();"`
    for `insert_after_once`, which matched five locations: lines 54, 84, 121, 127, and 155.
- Step 2:
  - The agent tried to continue the same Q-staging idea, but fell back to prose and omitted
    `candidate_transform`, causing planning validation failure.

Runtime change:

- Runtime commit `afaecd537b4e1967b379ccea018291ac21f94d3a`
  (`fix: surface transform anchor repair signal`) improves attempt-history follow-up.
- When the latest semantic `candidate_transform` fails materialization because a
  `replace_once`/`insert_*_once` anchor or match is ambiguous, the summary now adds a repair signal:
  - states the transform failed before compile;
  - preserves the full materialization error, including all matching line numbers;
  - includes the exact rejected `candidate_transform` JSON to repair;
  - tells the planner to repair anchors/matches with larger unique surrounding-code snippets and not
    restate the CUDA edit in prose.
- Live summary after the change now includes the full Q-staging repair signal, including:
  `matching start lines: 54, 84, 121, 127, 155` and the rejected transform JSON.

Verification:

- Focused tests:
  `.venv/bin/python -m pytest tests/test_evolve.py::test_summarize_attempt_history_requests_transform_anchor_repair tests/test_evolve.py::test_summarize_attempt_history_drops_compile_followup_after_preflight_rejection -q`
  passed, 2 tests.
- Live summary check:
  `.venv/bin/python - <<'PY' ... summarize_attempt_history(Path('attempts'), limit=6) ... PY`
  showed the new follow-up signal with the full ambiguous-anchor line list and rejected
  `candidate_transform` JSON.
- `.venv/bin/python -m pytest -q`: passed, 343 tests.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed in the runtime repo.
- Runtime commit `afaecd537b4e1967b379ccea018291ac21f94d3a` was pushed and fetch-verified on
  `coder-2011/avo-ampere`.

Decision:

- The live loop is now reaching meaningful semantic CUDA edits. The immediate blocker is not CUDA
  syntax or raw diffs; it is the planner's ability to repair structured transform anchors. The next
  bounded loop should test whether the explicit repair signal causes the agent to retry Q staging
  with unique anchors instead of prose-only planning.

## 2026-05-09 - Checkpoint 4.85: Score repaired Q-staging transform

Success criteria for this checkpoint:

- Test whether the anchor-repair signal causes the agent to repair the Q-staging transform.
- Score the compiled transform if the repair compiles.
- Record the performance result as search evidence.

Loop 1:

- Command:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 2 --loop-json attempts/loop_after_anchor_repair_signal.json --timeout-s 900 --env-file ../avo/.env.local`
- Result: no accepted candidate; `completed_steps=2`, `stopped_reason=max_steps`.
- Step 1 repaired the Q-staging transform with unique anchors and compiled successfully.
- Compile result:
  - no spills;
  - 40 registers;
  - 1 barrier;
  - 14016 bytes shared memory;
  - ptxas compile time about 109 ms.
- The compile-only patch was cleaned up as expected.
- Step 2 attempted to score the compiled transform but omitted `candidate_transform`; planning
  validation failed after three retries.

Runtime change after loop 1:

- Runtime commit `829f806df9f259fe151d822cffbb81499f6faef0`
  (`fix: clarify compiled transform scoring`) updates the prompt and follow-up signal.
- The follow-up signal now explicitly says compile-only patches are cleaned up before follow-up
  scoring, so a `no_edit` score would score the unmodified seed.
- It tells the planner to include the same `candidate_transform` again with
  `edit_mode=transform` and `candidate_patch=""`.
- Verification for that commit:
  - focused prompt/follow-up tests passed, 2 tests;
  - `.venv/bin/python -m pytest -q`: passed, 343 tests;
  - `.venv/bin/ruff check .`: passed;
  - `git diff --check`: passed;
  - pushed and fetch-verified on `coder-2011/avo-ampere`.

Loop 2:

- Command:
  `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 1 --loop-json attempts/loop_after_compiled_transform_score_hint.json --timeout-s 900 --env-file ../avo/.env.local`
- Result: no accepted candidate; `completed_steps=1`, `stopped_reason=max_steps`.
- The agent copied the pending Q-staging `candidate_transform` and scored the full target suite.
- Correctness: passed all 8 BF16 full-target cases.
- Geomean: `6.686302249012325` TFLOPS.
- Previous best: `7.777584666360881` TFLOPS.
- Gate: rejected for geomean regression.
- Per-case TFLOPS:
  - noncausal seq4096: `9.127144768610787`;
  - causal seq4096: `4.982410524364561`;
  - noncausal seq8192: `9.47602063778509`;
  - causal seq8192: `4.9226662010673605`;
  - noncausal seq16384: `9.231988362377242`;
  - causal seq16384: `4.778746088236129`;
  - noncausal seq32768: `9.257353783003765`;
  - causal seq32768: `4.610957708435425`.

Runtime knowledge update:

- Runtime commit `8d35171d6df859a690ae7ff4593bf2306e1047b6`
  (`docs: record q staging regression`) records the Q-staging result in
  `knowledge/ampere.md` and `knowledge/retrieval_claims.md`.
- The new claim is tested by retrieval query:
  `q_shared Q staging repaired anchors geomean 6.686302249012325 shared memory 14016`.
- Verification for that commit:
  - `.venv/bin/python -m pytest tests/test_knowledge.py -q`: passed, 36 tests;
  - `.venv/bin/python -m pytest -q`: passed, 344 tests;
  - `.venv/bin/ruff check .`: passed;
  - `git diff --check`: passed;
  - pushed and fetch-verified on `coder-2011/avo-ampere`.

Decision:

- The agent loop is now capable of repairing a structured CUDA transform, compiling it, and scoring
  the exact compiled transform after cleanup. That is real progress on the agent loop.
- The CUDA result itself is negative: isolated synchronous Q staging is correct but materially slower.
  This reinforces the earlier staging lesson: future work should not add standalone shared-memory
  Q tiles. Useful next directions are broader Q/K/V layout/copy-pipeline changes, K/V staging with
  real overlap, or a different work distribution that attacks the FA2 gap more directly.

## 2026-05-09 - Checkpoint 4.86: Add FA2 SM80 staging cue

Success criteria for this checkpoint:

- Do a fresh external source check after the repaired Q-staging regression.
- Add only useful information that changes future search behavior.
- Verify the new claim is retrievable by the planner.

Research:

- Exa query:
  `FlashAttention v2 Ampere SM80 forward kernel cp.async Q K V shared memory staging ldmatrix online softmax GitHub CUTLASS Dao-AILab implementation details`
- Primary source fetched:
  https://github.com/Dao-AILab/flash-attention/blob/3387de49/csrc/flash_attn/src/flash_fwd_kernel.h
- Useful finding:
  - FA2's SM80 forward kernel constructs `sQ`, `sK`, and `sV` shared-memory tensors with tiled copy
    layouts.
  - It can share Q/K shared memory with `Share_Q_K_smem`.
  - It can copy Q from shared memory into MMA register fragments when `Is_Q_in_regs` is active.
  - K and V are advanced through `cp.async` copy/fence/wait phases around QK/PV work.
- Interpretation:
  - The local `q_shared` regression is consistent with FA2's structure: "put Q in shared" alone is
    not the optimization. Q staging has to be coupled to swizzled shared layouts, smem-to-register
    copies, and the K/V async pipeline.

Runtime knowledge update:

- Runtime commit `d2afffbcafff682fe02dc94e1663965ed3e73f8e`
  (`docs: add fa2 sm80 staging cue`) updates:
  - `knowledge/ampere.md`;
  - `knowledge/retrieval_claims.md`;
  - `tests/test_knowledge.py`.
- New retrieval query:
  `FlashAttention SM80 Q in regs Share_Q_K_smem cp_async K V pipeline q_shared regression`.
- Why useful:
  - steers the planner away from another isolated synchronous `q_shared` transform;
  - points toward Q-in-register reuse, K/V async pipeline structure, or broader Q/K/V layout changes.

Verification:

- `.venv/bin/python -m pytest tests/test_knowledge.py -q`: passed, 37 tests.
- `.venv/bin/python -m pytest -q`: passed, 345 tests.
- `.venv/bin/ruff check .`: passed.
- `git diff --check`: passed in the runtime repo.
- Runtime commit `d2afffbcafff682fe02dc94e1663965ed3e73f8e` was pushed and fetch-verified on
  `coder-2011/avo-ampere`.

Decision:

- The next loop should be biased toward FA2-like dataflow coupling: Q register reuse and/or K/V
  async copy-pipeline work. A standalone Q shared-memory tile is now recorded as a confirmed
  negative family.

## 2026-05-09 - Checkpoint 4.87: Repair transform scope feedback and edit-channel logging

Success criteria for this checkpoint:

- Fix the recurring structured-transform failure without adding another narrow CUDA ban phrase.
- Keep the planner on semantic transforms rather than raw CUDA diffs.
- Verify with unit tests and a live evolve loop.

Loop before the fix:

- Runtime loop file: `attempts/loop_after_fa2_sm80_staging_cue.json`.
- Result: no accepted candidate.
- The planner moved in the right semantic direction by trying K staging, but the transform failed
  materialization because the anchor `__syncthreads();` matched five places.
- The failed insertion referenced `key_start`, so the repair needed to select an anchor inside the
  `for (int key_start = ...)` loop, not merely any unique sync point.

Runtime fixes:

- Runtime commit `72b5a42` (`fix: hint transform lexical scope repairs`):
  - adds a transform repair hint for inserts that reference loop-local `key_start` while anchoring
    outside that lexical context;
  - keeps the feedback phrased as scope repair, not a one-off CUDA phrase ban;
  - adds regression coverage in `tests/test_evolve.py`.
- Runtime commit `80ceb42` (`fix: preserve transform edit channel in attempts`):
  - keeps transform attempts' `decision.candidate_patch` as the planner's actual empty patch field;
  - records the orchestrator-generated unified diff separately as `materialized_patch`;
  - uses `materialized_patch` for cleanup and accepted lineage artifacts, so transform attempts stay
    recoverable without pretending the agent emitted a raw CUDA diff.

Live verification:

- Runtime loop file: `attempts/loop_after_scope_repair_hint.json`.
- Result: no accepted candidate.
- Step 1: planner validation rejected a repeated recorded unpatched MMA seed score.
- Step 2: the agent repaired the K-staging transform anchor using broader loop context, materialized
  the transform, and compiled it successfully:
  - no spills;
  - 40 registers;
  - 1 barrier;
  - 14016 bytes shared memory.
- The loop stopped at the two-step limit before scoring the compiled K-staging transform.

Verification:

- After `72b5a42`:
  - `.venv/bin/python -m pytest -q`: passed, 346 tests;
  - `.venv/bin/ruff check .`: passed;
  - `git diff --check`: passed.
- After `80ceb42`:
  - focused transform/channel tests: passed, 3 tests;
  - `.venv/bin/python -m pytest -q`: passed, 347 tests;
  - `.venv/bin/ruff check .`: passed;
  - `git diff --check`: passed.

Decision:

- The agent-side issue is improved in the intended direction: failed CUDA transforms now produce
  structural repair information the planner can act on, and attempt records no longer blur the
  transform interface with raw planner diffs.
- The next loop should either score the compiled K-staging transform or move to a materially
  different FA2-like dataflow change if the planner judges synchronous K staging alone too weak.

## 2026-05-09 - Checkpoint 4.88: Score K staging and sharpen next direction

Success criteria for this checkpoint:

- Run the agent loop after preserving the transform edit channel.
- Score the pending K-staging transform if the planner keeps that direction.
- Record the outcome as retrievable knowledge and use fresh external evidence to steer the next
  search direction.

Runtime loop:

- Runtime loop file: `attempts/loop_after_transform_channel_preserve.json`.
- Result: no accepted candidate.
- Steps 1 and 2 failed planner validation: the agent tried to score the compiled K-staging edit in
  prose without carrying the required `candidate_transform`.
- Step 3 repaired the decision contract:
  - `edit_mode="transform"`;
  - `candidate_patch=""`;
  - exact K-staging `candidate_transform`;
  - orchestrator-generated diff recorded separately as `materialized_patch`.
- The score passed all 8 full-target BF16 correctness cases but regressed:
  - geomean `4.16538030902376` TFLOPS;
  - best remained `7.777584666360881` TFLOPS;
  - gate reason: `candidate regressed geomean throughput`.

K-staging per-case TFLOPS:

- noncausal seq4096: `6.061454911980457`;
- causal seq4096: `3.038540148101854`;
- noncausal seq8192: `5.971483106041282`;
- causal seq8192: `2.9783210189160676`;
- noncausal seq16384: `5.797175125340377`;
- causal seq16384: `2.9231440595043137`;
- noncausal seq32768: `5.743468343189027`;
- causal seq32768: `2.8425023612542684`.

Research:

- Exa query:
  `FlashAttention-2 Ampere SM80 forward kernel cp.async K V shared memory tiled copy ldmatrix QK PV pipeline source code explanation`
- Useful sources:
  - NVIDIA CUTLASS CuTeDSL Ampere FlashAttention v2 example:
    https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py
  - Flash Attention from Scratch Part 2, Ampere building blocks:
    https://lubits.ch/flash/Part-2
- Useful finding:
  - productive Ampere K/V staging is a GMEM-to-SMEM-to-register pipeline:
    128-bit `cp.async` copies, swizzled/ldmatrix-compatible shared layouts,
    `LdMatrix8x8x16bOp` shared-to-register loads, MMA operand layouts, and overlap.
  - The local synchronous `k_shared` result is therefore an expected negative: it adds cooperative
    loads and an extra barrier without layout-aware `ldmatrix` use or overlap.

Runtime knowledge update:

- Runtime commit `73b49cd` (`docs: record k staging regression`) updates:
  - `knowledge/ampere.md`;
  - `knowledge/retrieval_claims.md`;
  - `tests/test_knowledge.py`;
  - follow-up scoring feedback in `avo/evolve.py` and `tests/test_evolve.py`.
- New retrieval queries:
  - `synchronous K shared memory staging regression geomean 4.16538030902376 k_shared key_start`;
  - `CUTLASS Ampere FlashAttention cp.async ldmatrix register pipeline 128-bit K V staging`.

Verification:

- Focused runtime tests:
  `.venv/bin/python -m pytest tests/test_evolve.py::test_summarize_attempt_history_requests_score_after_compile_only_transform tests/test_knowledge.py -q`
  passed, 40 tests.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 349 tests;
  - `.venv/bin/ruff check .`: passed;
  - `git diff --check`: passed.

Decision:

- Synchronous shared-memory staging for either Q or K is now a confirmed losing family on the
  current seed. Future work should not retry isolated `q_shared` or `k_shared` tiles.
- The next useful semantic direction is either a real Ampere copy/layout/register pipeline
  (`cp.async` plus ldmatrix-compatible shared layout and overlap) or a broader FA2-like work
  decomposition such as wider 128x128 tiling.

## 2026-05-09 - Checkpoint 4.89: Align compile diagnostics with score builds

Success criteria for this checkpoint:

- Run one more agent loop after adding the K-staging regression knowledge.
- Classify the next failure mode.
- Fix any root issue that is in the orchestrator/agent infrastructure rather than just adding a
  one-off CUDA ban.

Runtime loop:

- Runtime loop file: `attempts/loop_after_k_staging_regression_knowledge.json`.
- Result: no accepted candidate.
- Step 1: the planner proposed a broader synchronous K/V shared-memory staging transform.
  It compiled under standalone `avo compile` with:
  - no spills;
  - 40 registers;
  - 1 barrier;
  - 18112 bytes shared memory.
- Step 2: the planner correctly scored the same transform with `edit_mode="transform"` and
  `candidate_patch=""`, but score-time Torch extension compilation failed on the first case:
  `__nv_bfloat16(0.0f)` is invalid when `__CUDA_NO_BFLOAT16_CONVERSIONS__` is defined.
- This exposed an infrastructure mismatch: standalone compile diagnostics did not use the same
  CUDA half/BF16 disabling macros as Torch extension builds.

Runtime fix:

- Runtime commit `8536974` (`fix: align compile diagnostics with score builds`) updates
  `avo compile` to pass the Torch extension CUDA defines:
  - `__CUDA_NO_HALF_OPERATORS__`;
  - `__CUDA_NO_HALF_CONVERSIONS__`;
  - `__CUDA_NO_BFLOAT16_CONVERSIONS__`;
  - `__CUDA_NO_HALF2_OPERATORS__`.
- The fix also records the compile/score contract in:
  - `knowledge/ampere.md`;
  - `knowledge/retrieval_claims.md`;
  - `tests/test_knowledge.py`.
- New retrieval query:
  `compile score torch extension CUDA_NO_BFLOAT16_CONVERSIONS __nv_bfloat16 constructor`.

Verification:

- `uv run --extra cuda python -m avo compile --source candidates/cuda_mma_attention/attention_kernel.cu --out-dir build/strict_compile_contract --timeout-s 120`
  passed with the stricter flags and the accepted seed still compiled at 40 registers, 1 barrier,
  and 9920 bytes shared memory.
- Focused runtime tests:
  `.venv/bin/python -m pytest tests/test_compile.py tests/test_knowledge.py -q`
  passed, 42 tests.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 350 tests;
  - `.venv/bin/ruff check .`: passed;
  - `git diff --check`: passed.

Decision:

- Compile-only diagnostics are now a better proxy for score-time extension builds. This should
  prevent another false compile success for BF16 constructor/conversion code that scoring later
  rejects.
- The synchronous K/V staging family remains unattractive. The next loop should move away from
  immediate shared staging toward a real async/copy-layout/register-pipeline step or a wider
  FA2-like tile/work decomposition.

## 2026-05-09 - Checkpoint 4.90: Post compile-contract loop

Success criteria for this checkpoint:

- Run a short loop after aligning `avo compile` with score-time extension flags.
- Check whether the next failure appears at compile/score time or earlier in planning.

Runtime loop:

- Runtime loop file: `attempts/loop_after_compile_contract_fix.json`.
- Result: no accepted candidate.
- Step 1 failed planner validation because the returned decision omitted required fields:
  `expected_effect`, `risk`, and `next_command`.
- Step 2 failed planner validation because the agent proposed increasing `kThreads` from 128 to 256
  in prose without a `candidate_transform` or patch payload.
- No compile or score command ran in this loop, so it did not exercise the stricter compile flags.

Decision:

- The compile-contract fix remains valid and verified, but this loop surfaced planner-format drift
  rather than a kernel result.
- The recent-attempt supervisor signal now says the last five attempts produced no accepted
  candidate. The next useful action is a strategy reset toward a different Ampere optimization
  family, not another synchronous shared-memory staging attempt or thread-count prose edit.

## 2026-05-09 - Checkpoint 4.91: Improve transform recovery and record wider-tile invariant

Success criteria for this checkpoint:

- Improve recovery for structured planner outputs without broadening raw CUDA diffs.
- Verify the agent can move past prior edit-channel failures.
- Record any new CUDA structural lesson from the next loops.

Runtime fixes:

- Runtime commit `056a62f` (`fix: infer constant retune transforms`):
  - parser recovery now infers `set_constexpr_int` from natural constant-retune verbs such as
    "increase", "decrease", and "retune", not only "change/set/update";
  - this turns prose like "increase kThreads from 128 to 256 in candidates/.../attention_kernel.cu"
    into a structured transform instead of a missing-edit-payload failure.
- Runtime commit `7ebea08` (`fix: accept batch default transform path`):
  - accepts `op=batch` with a top-level default `path` and fills that path into steps that omit it;
  - keeps the Anthropic tool schema slim by advertising compact `steps_json` rather than a nested
    `steps` array, after a loop proved the nested schema caused `Schema is too complex`.
- Runtime commit `e51adcd` (`docs: record wider tile invariant`):
  - records the `kTile=32` structural lesson in `knowledge/ampere.md` and
    `knowledge/retrieval_claims.md`;
  - adds retrieval coverage in `tests/test_knowledge.py`.

Runtime loops:

- `attempts/loop_after_constant_retune_inference.json`:
  - no accepted candidate;
  - exposed the top-level batch `path` shorthand failure.
- `attempts/loop_after_batch_path_recovery.json`:
  - failed before planning because the nested `steps` schema was too complex for Anthropic.
- `attempts/loop_after_batch_path_recovery_schema_fix.json`:
  - schema request succeeded again;
  - agent proposed a real wider-query-tile transform (`kTile=32`) instead of another synchronous
    shared-memory staging retry;
  - materialization failed because one row-loop `replace_once` matched two locations.
- `attempts/loop_after_tile32_anchor_repair_signal.json`:
  - agent tried to repair the `kTile=32` transform with larger anchors;
  - materialization still failed on one stale anchor;
  - follow-up planning then self-rejected the family because existing MMA fragments only cover the
    first 16 rows of a widened 32-row tile.

Verification:

- Focused runtime tests:
  `.venv/bin/python -m pytest tests/test_agent.py::test_parse_variation_decision_infers_increase_constexpr_transform_from_edit tests/test_agent.py::test_parse_variation_decision_applies_batch_default_path tests/test_knowledge.py -q`
  passed, 43 tests.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 353 tests;
  - `.venv/bin/ruff check .`: passed;
  - `git diff --check`: passed.

Decision:

- The agent interface is better: natural constant retunes and common batch-path shorthand now recover
  into structured transforms instead of failing at the edit-channel boundary.
- Widening the MMA query tile is still a plausible direction, but not as a constants/buffer-only
  transform. A correct wider tile needs explicit second 16-row sub-tile MMA, softmax, and output
  dataflow. Otherwise rows are uncovered.

## 2026-05-09 - Checkpoint 4.92: Add edit self-repair and fix score-worker environment

Success criteria for this checkpoint:

- Let the evolve loop repair its own executable edit failures instead of only recording them.
- Keep CUDA evolution on structured semantic transforms rather than raw CUDA diffs or prose edits.
- Run a longer loop budget (`--max-steps 40`, `--timeout-s 2400`, `--compile-repair-attempts 6`) and
  verify whether scoring and repair work on realistic long-sequence lanes.

Runtime fixes:

- Added compile self-repair in `avo/cli.py`/`avo/evolve.py`:
  - compile failures from applied candidate edits are reverted;
  - compiler stderr/stdout plus the failed edit payload are fed back as an immediate repair request;
  - the repair decision must contain a revised executable edit payload, not a no-edit retry.
- Added structured-transform materialization self-repair:
  - repairable anchor/match failures now trigger an immediate repair request with the rejected
    transform JSON and materialization error;
  - unchanged payload retries and prose-only repairs are rejected.
- Hardened pending compile-only transform follow-up:
  - score-shaped planner outputs now get the exact pending `candidate_transform` attached even when
    the planner omits or malforms the field;
  - compile-only transforms with failed cleanup are no longer treated as valid pending transforms.
- Fixed candidate scoring environment:
  - score workers now prepend the active Python environment's `bin` directory to `PATH`, so PyTorch
    can find `.venv/bin/ninja`;
  - score failures caused by infrastructure errors such as missing Ninja are classified as
    `score_environment_error` instead of CUDA correctness failures.
- Added small semantic transform recovery:
  - "thread count" retunes map to `set_constexpr_int(..., name="kThreads")`;
  - "changing/widening/narrowing kTile from X to Y" is parsed as a constant transform rather than a
    missing-payload prose edit.

Runtime loop observations:

- `attempts/2026-05-09T18-58-57-00-00.json`:
  - K shared-memory staging compiled cleanly (`14016 bytes smem`, `40 registers`, no spills);
  - cleanup succeeded, leaving the transform pending for score.
- `attempts/2026-05-09T19-04-13-00-00.json`:
  - the pending K-staging transform was scored before the score-worker environment fix;
  - it was falsely marked `all_correct=false` because the first case failed with
    `Ninja is required to load C++ extensions`.
- Manual score after the environment fix:
  - `avo score --backend candidate --candidate candidates/cuda_mma_attention_seed.py --seq-lens 4096 ...`
    passed with `all_correct=true`;
  - the two 4096 cases measured `11.39 TFLOPS` noncausal and `5.39 TFLOPS` causal, geomean
    `7.83 TFLOPS`.
- `attempts/2026-05-09T19-19-41-00-00.json` then
  `attempts/2026-05-09T19-22-08-00-00.json`:
  - the loop compiled a repaired K-only staging transform and scored it on
    `4096,8192,16384,32768`;
  - the score was now valid (`all_correct=true`, no environment errors);
  - geomean was `4.16 TFLOPS`, rejected only for throughput regression against best candidate
    geomean `7.78 TFLOPS`.
- `attempts/2026-05-09T19-30-33-00-00.json`:
  - compile self-repair fired on a row-major K WMMA idea;
  - the repair correctly identified that Ampere BF16 WMMA does not support that row-major `matrix_b`
    shape/layout contract and pivoted back toward shared K staging;
  - the next repair failed materialization on an ambiguous anchor and then dropped the payload.
- The loop still sometimes falls back to prose-only CUDA edits or unsupported profiler requests.
  These are now classified and rejected earlier, but the planner still needs better supported
  diagnostic tools or stronger transform repair behavior to avoid repeated planning churn.

Verification:

- Focused runtime tests for the new behavior passed:
  - compile self-repair;
  - structured-transform materialization repair;
  - pending transform payload normalization;
  - score-worker Ninja PATH handling;
  - score environment-error classification;
  - thread-count and `kTile` semantic constant inference.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 363 tests;
  - `.venv/bin/ruff check .`: passed;
  - `git diff --check`: passed.

Decision:

- The loop is now genuinely repairing some of its own edit failures. It no longer just records a
  compiler error and waits for a human patch.
- The current best validated kernel is still the prior seed lane; K shared-memory staging is correct
  but slower on the long-sequence target lane.
- The next root issue is planner behavior around unsupported diagnostics and prose-only repairs, not
  CUDA compilation or score infrastructure.

## 2026-05-10 - Checkpoint 4.93: Bound profiler diagnostics and score pending transforms

Success criteria for this checkpoint:

- Add a bounded profiler diagnostic without letting Nsight/CUPTI failures hang the evolve loop.
- Prevent unsupported profiler requests from consuming repeated loop steps in the Thunder runtime.
- Make pending compile-only structured transforms reliably carry forward into score steps, including
  when the planner emits placeholder prose in `candidate_patch`.
- Keep CUDA preflight structural checks from overfitting: hard-reject only clearly scalar BF16
  async-copy calls, and let broader async-copy hypotheses reach compile/repair.
- Run a higher-budget evolve loop and verify that it uses compile/score/self-repair instead of
  stalling on profile or losing the pending transform.

Research consulted:

- NVIDIA Nsight Compute CLI documentation:
  `https://docs.nvidia.com/nsight-compute/NsightComputeCli/`
- NVIDIA Nsight Compute Profiling Guide:
  `https://docs.nvidia.com/nsight-compute/ProfilingGuide/`
- Local runtime observation: `ncu` exists, but in this container the Thunder CUDA shim aborts CUPTI
  queries with an unsupported-runtime error. The implementation therefore treats profiling as a
  best-effort diagnostic and reports structured unavailability before launch when the Thunder marker
  is present.

Runtime fixes:

- Added `avo profile`:
  - wraps the isolated `worker-score` command in Nsight Compute;
  - supports candidate-only profiling with score-shaped workload flags;
  - emits structured JSON with profiler settings, score payload when available, and classified
    profiler errors such as `ncu_not_found`, `profiler_permission`,
    `profiler_unsupported_runtime`, `no_kernels_profiled`, and `timeout`;
  - caps the profiler subprocess at 120 seconds even if a larger score timeout is requested.
- Added planner validation for `avo profile`:
  - only candidate profiling is allowed;
  - profile commands require profiler intent, not correctness/timing intent;
  - this Thunder-backed runtime rejects profile decisions during validation so the planner must repair
    the decision before a loop step is spent.
- Fixed pending-transform score follow-up:
  - the payload normalizer now attaches the latest pending compile-only `candidate_transform` even
    if the planner emits a non-diff placeholder in `candidate_patch`;
  - this keeps the exact structured edit attached to the score command instead of scoring the
    unmodified seed or failing as a prose-only edit.
- Relaxed the async-copy granularity preflight:
  - it still rejects clearly scalar BF16 async-copy calls such as
    `__pipeline_memcpy_async(..., sizeof(__nv_bfloat16))`;
  - it no longer rejects broader grouped-size async-copy hypotheses solely because they mention
    `sizeof(__nv_bfloat16)`, so compile/self-repair can classify those attempts.

Long-loop observations:

- `attempts/loop_higher_budget_20260509T2010Z.json` ran with:
  - `--max-steps 8`;
  - `--timeout-s 1800`;
  - `--compile-repair-attempts 4`.
- The loop completed all 8 steps, accepted no candidate, and stopped at `max_steps`.
- Step 1 scored the pending K shared-memory staging transform:
  - `all_correct=true`;
  - geomean `4.1946 TFLOPS`;
  - rejected as a throughput regression versus best candidate geomean `7.7776 TFLOPS`.
- Steps 2, 5, and 6 compiled async/shared K-staging variants successfully, with cleanup after
  rejected compile-only checks.
- Step 7 scored another cooperative K shared-memory staging variant:
  - `all_correct=true`;
  - geomean `4.1529 TFLOPS`;
  - rejected as a throughput regression.
- Step 8 still ended in a prose-only planning failure after three retries. The loop is now past the
  profile trap and can score pending transforms, but planner discipline around executable
  `candidate_transform` payloads still needs more work.

Verification:

- Runtime full suite:
  - `.venv/bin/python -m pytest -q`: passed, 372 tests;
  - `.venv/bin/ruff check .`: passed;
  - `git diff --check`: passed.
- Focused coverage added for:
  - `avo profile` command construction and classified profiler failures;
  - profile validation and unsupported-runtime rejection;
  - profile attempt-history classification;
  - pending-transform normalization when the planner emits a placeholder patch string;
  - relaxed async-copy granularity preflight.

Decision:

- `avo profile` is useful as a bounded diagnostic interface, but not as a required part of the
  current Thunder runtime search loop.
- The loop is closer to the desired behavior: compile-only semantic transforms can be scored without
  human repair, and unsupported profiling no longer consumes repeated long steps.
- The next root issue is still planner payload discipline: after several rejected/regressed steps,
  it can regress to prose-only transform descriptions. That should be handled by improving the
  search loop and repair prompt, not by adding a larger brittle ban list.

## 2026-05-10 - Checkpoint 4.94: Require explicit transform field in agent tool output

Success criteria for this checkpoint:

- Improve planner payload discipline without adding a CUDA-specific phrase ban.
- Keep the parser backward-compatible with older JSON payloads and tests.
- Verify that the real Anthropic `agent-plan` path accepts the stricter tool schema.

Research consulted:

- Exa search for current production tool-calling reliability patterns found consistent guidance:
  keep schema validation as the gate, return structured validation errors to the model, cap retries,
  and improve schema/data-contract clarity before tuning prompts. The relevant pattern for this
  checkpoint is to make required argument fields explicit and use validation feedback rather than
  executing malformed tool calls.

Runtime fix:

- Updated the variation decision tool schema so `candidate_transform` is always a required field:
  - transform mode must provide the structured object;
  - no-edit and legacy-patch modes must set `candidate_transform` to `null`;
  - local parsing still defaults a missing field to `None` so older saved JSON and unit tests remain
    compatible.
- Updated the planner prompt to say that `candidate_transform` must always be present and should be
  `null` outside transform mode.

Verification:

- Live Anthropic schema smoke:
  - `avo agent-plan --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts
    --attempt-limit 8 --env-file ../avo/.env.local`;
  - succeeded and returned an explicit structured `candidate_transform` batch for a Q/K shared-memory
    staging compile check.
- Runtime full suite:
  - `.venv/bin/python -m pytest -q`: passed, 372 tests;
  - `.venv/bin/ruff check .`: passed;
  - `git diff --check`: passed.

Decision:

- This is a better root fix than another reactive guard: it tightens the tool contract so the model
  has to make the executable edit channel explicit every turn, while keeping the same validation and
  retry behavior for bad semantic payloads.

## 2026-05-10 - Checkpoint 4.95: Relax async-copy granularity preflight

Success criteria for this checkpoint:

- Keep hard preflights for structural invariants, not CUDA phrase-level preferences.
- Let coherent async-copy hypotheses reach compile and compile-repair even when the copy granularity
  looks weak.
- Preserve the raw CUDA diff ban and the structured transform interface.
- Verify the planner validation and tests after the change.

Loop evidence before the change:

- A four-step evolve-loop with the explicit `candidate_transform` schema ran to `max_steps` with no
  accepted candidate.
- Step 1 compiled a structured Q shared-memory staging transform for the MMA kernel:
  - ptxas used 44 registers, 1 barrier, and 14016 bytes shared memory;
  - cleanup after the compile-only check succeeded.
- Step 2 scored that pending transform on the realistic A6000 target lane:
  - `all_correct=true`;
  - geomean `6.6941 TFLOPS`;
  - rejected versus best geomean `7.7776 TFLOPS`.
- Step 3 compiled a repaired V async-pipeline staging transform after a failed first attempt. This
  confirmed the compile self-repair path can produce a compiling CUDA candidate without manual edit.
- Step 4 scored the repaired V async-pipeline transform:
  - `all_correct=true`;
  - geomean `6.7332 TFLOPS`;
  - rejected versus best geomean `7.7776 TFLOPS`.

Runtime fix:

- Removed the always-on `async_copy_granularity` structural preflight. Scalar BF16
  `__pipeline_memcpy_async` granularity is now guidance for the planner and repair loop, not a
  standalone pre-compile rejection.
- Kept stronger structural preflights in place:
  - raw CUDA diffs are still rejected in favor of `candidate_transform`;
  - invalid async pipeline stage lifecycle is still rejected;
  - templated `__pipeline_wait_prior<...>` API misuse is still rejected;
  - no-effect wrapper/stub/dataflow checks remain active.
- Updated the agent context and CUDA knowledge so vector-group async copies remain the preferred
  Ampere direction, but the loop can validate concrete async-copy attempts through compile/score.

Verification:

- Focused runtime tests:
  - scalar BF16 async-copy materialized CUDA patches now pass structural preflight when CUDA source
    edits are explicitly allowed for materialized transforms;
  - validation feedback now says to prefer vector-group async-copy dataflow while letting
    compile/repair validate concrete errors.
- Runtime full suite:
  - `.venv/bin/python -m pytest -q`: passed, 372 tests;
  - `.venv/bin/ruff check .`: passed;
  - `git diff --check`: passed.

Decision:

- The async-copy granularity rule was too strict as a hard gate. It encoded a useful CUDA preference
  as a rejection before the agent could use compiler feedback or repair.
- The better boundary is structural: reject edits that cannot be represented, violate known
  lifecycle/API invariants, or have no executable dataflow. Let weak but coherent CUDA hypotheses
  compile, fail, and self-repair.

## 2026-05-10 - Checkpoint 4.96: Verify relaxed async-copy loop behavior

Success criteria for this checkpoint:

- Run a longer evolve-loop after removing the scalar BF16 async-copy hard preflight.
- Confirm that async-copy attempts are routed through compile and compile-repair instead of being
  blocked before compile.
- Confirm that rejected candidates clean up their source edits and do not dirty the runtime repo.
- Do not add a new phrase-level guard just because candidates fail correctness.

Run:

```bash
timeout 14400 .venv/bin/python -m avo evolve-loop \
  --lineage ./lineage \
  --knowledge knowledge/ampere.md \
  --cwd . \
  --env-file ../avo/.env.local \
  --timeout-s 2400 \
  --max-steps 8 \
  --compile-repair-attempts 4 \
  --attempts-dir ./attempts \
  --attempt-limit 24 \
  --loop-json attempts/loop_relaxed_async_preflight_20260510T0317Z.json
```

Result:

- Exit code `2`: no accepted candidate.
- `completed_steps=8`, `stopped_reason=max_steps`.
- No runtime source edits remained after cleanup.

Step summary:

- Step 1 compiled a repaired K double-buffer async-copy transform:
  - `ok=true`;
  - ptxas reported 40 registers, 1 barrier, and 18112 bytes shared memory;
  - one compile-repair attempt preceded the successful compile.
- Step 2 scored that transform:
  - `all_correct=false`;
  - geomean `0`;
  - gate rejected it for correctness failure.
- Step 3 compiled a repaired K async staging transform:
  - `ok=true`;
  - one compile-repair attempt preceded the successful compile.
- Step 4 scored that transform:
  - `all_correct=false`;
  - geomean `0`;
  - gate rejected it for correctness failure.
- Step 5 compiled a repaired Q async-copy staging transform:
  - `ok=true`;
  - repair changed the inline PTX global pointer operand from a 32-bit `unsigned`
    value to `unsigned long long` for the `l` constraint.
- Step 6 scored that transform:
  - `all_correct=false`;
  - geomean `0`;
  - gate rejected it for correctness failure.
- Step 7 compiled another repaired K async-copy staging transform:
  - `ok=true`;
  - repair again fixed the `cp.async` global pointer operand width.
- Step 8 scored that transform:
  - `all_correct=false`;
  - geomean `0`;
  - score cases reported CUDA unknown/misaligned-address errors;
  - gate rejected it for correctness failure.

What this proved:

- The relaxed async-copy granularity boundary is working as intended. The planner generated async
  hypotheses, the materializer applied them, compile errors were repaired inside the agent loop, and
  scoring rejected bad runtime behavior without human intervention.
- The old scalar async-copy hard preflight would have blocked parts of this search before compile.
  Letting the candidates reach compile exposed more useful concrete failures, especially the
  64-bit pointer operand requirement for inline PTX `cp.async`.
- The next issue is not another ban list. These async candidates are semantically weak: they stage
  K or Q with barriers/commit/wait patterns that compile but do not preserve a correct initialized
  dataflow. The search loop should use the correctness failures as attempt history and move toward
  better staging semantics, not add a new hard rule for every misaligned-address variant.

Decision:

- Keep the relaxed preflight.
- Record the loop as evidence that compile self-repair is functioning for concrete CUDA compile
  errors.
- Do not promote these correctness failures to a hard preflight yet; the repeated class is broad
  (`correctness_failed` / runtime misaligned address) and needs a better semantic transform family,
  not another phrase-specific rejection.

## 2026-05-10 - Checkpoint 4.97: Add immediate correctness repair

Success criteria for this checkpoint:

- Extend the existing edit-repair loop without changing the candidate edit interface.
- Repair only score correctness failures, not throughput regressions.
- Revert the failed patch before asking for the repair, matching compile/materialization repair.
- Keep the repair prompt structural: ask for a revised executable edit that addresses the violated
  invariant, not a phrase-specific guard.
- Verify with focused tests plus the full runtime test suite.

Runtime fix:

- Added `attempt_has_repairable_correctness_failure`:
  - requires an applied candidate edit;
  - requires a successful command result with a score payload;
  - requires `all_correct=false`;
  - ignores throughput-only gate rejections.
- Added correctness failure summary helpers so the repair prompt includes the classified failure and
  score error text.
- Extended the evolve repair router:
  - materialization failures still ask for anchor/match repair;
  - compile failures still ask for compiler-output repair;
  - correctness failures now ask for an immediate semantic repair after cleanup.
- The correctness repair prompt points the agent at initialized dataflow, alignment, bounds,
  masking, synchronization, and accumulation semantics. It does not add a new CUDA phrase ban.
- README now documents that failed materialization, compile, or score-correctness edits can trigger
  immediate revised-edit repair.

Verification:

- Focused repair tests:
  - compile repair still works;
  - transform materialization repair still works;
  - new correctness repair test simulates a score payload with `all_correct=false`, verifies that
    the failed patch is reverted, verifies the repair prompt includes the score error, and verifies
    the repaired score becomes the final accepted attempt.
- Runtime full suite:
  - `.venv/bin/python -m pytest -q`: passed, 373 tests;
  - `.venv/bin/ruff check .`: passed;
  - `git diff --check`: passed.

Decision:

- This closes the most obvious remaining self-repair gap from the relaxed async-copy loop. Compile
  repair was already functioning; now a candidate that compiles but fails correctness can also get
  one or more immediate revised executable edits before the top-level loop moves on.
- Throughput regressions remain normal gate rejections. Repairing those immediately would blur
  search with local debugging and is not the same class as a correctness violation.

## 2026-05-10 - Checkpoint 4.98: Accept Q-fragment register reuse candidate

Success criteria for this checkpoint:

- Run a bounded live evolve-loop after adding correctness-repair support.
- Accept a candidate only if it passes the full realistic A6000 BF16 score lane.
- Preserve and commit the accepted runtime candidate source.
- Record the accepted lineage evidence and update the runtime knowledge base.

Run:

```bash
timeout 10800 .venv/bin/python -m avo evolve-loop \
  --lineage ./lineage \
  --knowledge knowledge/ampere.md \
  --cwd . \
  --env-file ../avo/.env.local \
  --timeout-s 2400 \
  --max-steps 4 \
  --compile-repair-attempts 3 \
  --attempts-dir ./attempts \
  --attempt-limit 28 \
  --loop-json attempts/loop_correctness_repair_20260510T0402Z.json
```

Result:

- Exit code `0`.
- `completed_steps=3`, `stopped_reason=accepted`.
- Lineage accepted commit: `c0b53dd evolve: accept candidate`.
- Runtime accepted source: `candidates/cuda_mma_attention/attention_kernel.cu`.

Step summary:

- Step 1 failed planning validation after the planner retried unsupported `avo profile` requests in
  the Thunder runtime. No command executed.
- Step 2 compiled a Q-register-reuse transform after two repair attempts:
  - initial repair attempt failed compile because `q_frags` was scoped inside the warp guard and then
    used outside that lexical scope;
  - the repaired transform declared `q_frags[8]` outside the key-tile loop, loaded the eight Q WMMA
    fragments once, and reused them inside the QK loop;
  - ptxas reported 64 registers, 1 barrier, 9920 bytes shared memory, and no spills.
- Step 3 scored the compiled transform on the realistic target lane:
  - all 8 BF16 cases passed correctness;
  - geomean `8.960753680686471` TFLOPS;
  - prior best geomean was `7.777584666360881` TFLOPS;
  - gate accepted the candidate.

Per-case accepted score:

- seq4096: noncausal `12.5894` TFLOPS, causal `6.3634` TFLOPS.
- seq8192: noncausal `12.8563` TFLOPS, causal `6.3876` TFLOPS.
- seq16384: noncausal `12.6888` TFLOPS, causal `6.2818` TFLOPS.
- seq32768: noncausal `12.7451` TFLOPS, causal `6.2195` TFLOPS.

Verification:

- Runtime loop score: all accepted cases correct.
- Runtime full suite after applying the accepted source:
  - `.venv/bin/python -m pytest -q`: passed, 373 tests;
  - `.venv/bin/ruff check .`: passed;
  - `git diff --check`: passed.

Decision:

- Q-fragment register reuse is now the best local MMA candidate direction.
- The accepted improvement is not from shared memory or async copy; it removes repeated global Q
  fragment loads by keeping Q in WMMA fragments across the key-tile loop.
- Future search should preserve this Q-in-register reuse while exploring K/V traffic, PV/softmax
  scheduling, or broader FA2-like pipelining.

## 2026-05-10 - Checkpoint 4.99: Surface unavailable profiling before planning

Success criteria for this checkpoint:

- Avoid wasting agent planning retries on `avo profile` when the current runtime already marks
  Nsight/CUPTI profiling as unsupported.
- Keep profile validation in place for safety; this is planner context, not a replacement for the
  command validator.
- Verify prompt/context construction with tests.

Runtime fix:

- `build_repo_context` now checks the same Thunder runtime marker used by profile validation.
- When the marker exists, the bounded-command context says `avo profile is unavailable in this
  runtime`.
- The profile guidance line now says not to choose `avo profile` under the Thunder-backed execution
  environment and to use `score` or compile a `candidate_transform` instead.
- When the marker is absent, the normal `avo profile` guidance remains available.

Verification:

- Focused tests:
  - repo context lists local candidates as before;
  - repo context marks profile unavailable when a mocked Thunder marker exists.
- Runtime checks:
  - `.venv/bin/ruff check .`: passed;
  - `git diff --check`: passed.
- A full runtime suite had just passed after the accepted candidate checkpoint:
  - `.venv/bin/python -m pytest -q`: passed, 374 tests after adding the focused context test.

Decision:

- This addresses the planning-validation failure seen in the accepted-candidate loop without adding
  another reactive ban. The runtime fact is now in the planner's first context, and the existing
  validator remains the hard boundary if the planner still asks for profile.

## 2026-05-10 - Checkpoint 5.00: Post-Q-reuse V-staging loop

Success criteria for this checkpoint:

- Refresh the Ampere/FA2 context online before the next live loop.
- Verify that the new profile-unavailable planner context avoids the unsupported `avo profile`
  failure.
- Continue search from the accepted Q-fragment-reuse baseline.
- Record negative and pending V-staging evidence without adding a new guard.

Research consulted:

- Exa found NVIDIA CUTLASS's current Ampere FA2 CuTe example. The relevant grounding is that a real
  Ampere FA2 structure combines 128-bit `cp.async` Q/K/V movement, swizzled shared layouts, tensor
  cores, a register pipeline, and online-softmax fusion.
- Exa also surfaced Dao-AILab's SM80 mainloop showing the `Q_in_regs` pattern coupled to K/V
  `cp_async` fence/wait stages around QK, softmax/rescale, and PV.

Runtime knowledge update:

- Added a note that the accepted `q_frags[8]` direction should be preserved.
- Future semantic moves should target K/V stage scheduling, V/PV movement, softmax/PV overlap, or a
  broader FA2-like pipeline rather than isolated synchronous shared tiles.

Run:

```bash
timeout 10800 .venv/bin/python -m avo evolve-loop \
  --lineage ./lineage \
  --knowledge knowledge/ampere.md \
  --cwd . \
  --env-file ../avo/.env.local \
  --timeout-s 2400 \
  --max-steps 4 \
  --compile-repair-attempts 3 \
  --attempts-dir ./attempts \
  --attempt-limit 32 \
  --loop-json attempts/loop_post_q_reuse_20260510T0416Z.json
```

Result:

- Exit code `2`.
- `completed_steps=4`, `stopped_reason=max_steps`.
- No accepted candidate.
- No unsupported `avo profile` attempt occurred; the profile context fix worked.

Step summary:

- Step 1 compiled a V single-stage shared-memory staging transform after one repair.
- Step 2 scored that transform:
  - `all_correct=true`;
  - geomean `4.606980002471371` TFLOPS;
  - rejected versus best geomean `8.960753680686471`.
- Step 3 failed planning validation because the planner asked `avo score` for profiler-style
  evidence. This is a remaining prompt-discipline issue, but the validator correctly blocked it.
- Step 4 compiled a V double-buffer `cp.async` staging transform:
  - ptxas reported 64 registers, 1 barrier, 18112 bytes shared memory, and no spills;
  - cleanup succeeded because the step was compile-only and not accepted;
  - the loop stopped before scoring it.

Decision:

- Synchronous single-stage V staging is negative evidence: correct but much slower.
- The V double-buffer async candidate is pending evidence, not accepted evidence. It should be scored
  in a follow-up only if the pending-transform logic still carries the exact transform.
- Do not add a guard for V staging; the useful distinction is synchronous/no-overlap versus a real
  async pipeline.

## 2026-05-10 - Checkpoint 5.01: Clarify score versus profile before planning

Success criteria for this checkpoint:

- Reduce wasted planner turns where the agent asks `avo score` for profiler-style evidence.
- Keep this as command semantics, not a new reactive CUDA ban.
- Verify the planner context and existing hard validation still agree.

Runtime fix:

- `build_repo_context` now states that `avo score` reports correctness validation, timing samples,
  and TFLOPS only.
- The same context states that score does not report bandwidth, occupancy, scheduler stalls,
  instruction mix, or tensor-core utilization.
- Existing validation still rejects score decisions that claim profiler evidence; this change
  surfaces the contract before the planner chooses the command.

Verification:

- Focused runtime tests:
  - `.venv/bin/python -m pytest tests/test_agent.py -q`: passed, 185 tests;
  - `.venv/bin/ruff check avo/agent.py tests/test_agent.py`: passed;
  - `git diff --check`: passed.

Decision:

- The current change is a planner-information fix. It does not change CUDA preflight policy or add a
  new attempt-specific phrase ban.

## 2026-05-10 - Checkpoint 5.02: Follow-up loop after score/profile context fix

Success criteria for this checkpoint:

- Continue from the accepted Q-fragment-reuse MMA baseline.
- Score the pending V double-buffer async candidate if the attempt-history follow-up keeps it alive.
- Observe whether the planner still confuses score and profiler evidence after the context fix.
- Record accepted, rejected, and pending evidence without adding a new guard.

Run:

```bash
timeout 14400 .venv/bin/python -m avo evolve-loop \
  --lineage ./lineage \
  --knowledge knowledge/ampere.md \
  --cwd . \
  --env-file ../avo/.env.local \
  --timeout-s 3600 \
  --max-steps 6 \
  --compile-repair-attempts 3 \
  --attempts-dir ./attempts \
  --attempt-limit 40 \
  --loop-json attempts/loop_score_pending_v_async_20260510T0421Z.json
```

Result:

- Exit code `2`.
- `completed_steps=6`, `stopped_reason=max_steps`.
- No accepted candidate.
- No `avo score` request for profiler-only evidence occurred after the context fix.

Step summary:

- Step 1 scored the pending V double-buffer `cp.async` transform:
  - `all_correct=false`;
  - geomean `0`;
  - rejected for correctness failure with CUDA unknown errors.
- Step 2 produced an invalid key-tile-32 transform. The decision itself admitted the transform did
  not update the PV side and would likely fail; the step did not execute a command.
- Step 3 compiled an isolated K shared-memory staging transform.
- Step 4 scored that K staging transform:
  - `all_correct=true`;
  - geomean `4.377717248710737` TFLOPS;
  - rejected versus best geomean `8.960753680686471`.
- Step 5 compiled a vectorized 16-byte K `cp.async` staging repair:
  - ptxas reported 64 registers, 1 barrier, 14016 bytes shared memory, and no spills;
  - cleanup succeeded because the step was compile-only and not accepted.
- Step 6 failed planning validation:
  - the correctness-repair decision repeated the failed edit payload unchanged;
  - cleanup succeeded.

Decision:

- The loop is now doing the right broad kind of work: compile, score, reject, and clean up concrete
  semantic transforms rather than letting failures leak into the candidate source.
- The evidence is negative for isolated K/V shared staging. Correct single-stage K or V staging is
  much slower than the Q-fragment-reuse best.
- The next useful CUDA direction should either score a genuinely repaired K async path with correct
  per-tile offsets and stage lifetime, or move to a wider FA2-like scheduling/dataflow transform.

## 2026-05-10 - Checkpoint 5.03: Keep pending-transform autofill out of repairs

Success criteria for this checkpoint:

- Fix the repair-loop issue observed in checkpoint 5.02, where a correctness-repair request could
  reattach the same pending compiled transform instead of requiring a revised edit.
- Keep pending-transform autofill available for normal top-level follow-up scoring.
- Verify rejected repair edits still clean up the worktree.

Runtime fix:

- Immediate compile/materialization/correctness repair requests now call the agent without the
  pending-transform payload normalizer.
- Top-level planning still uses the normalizer, so "score the previously compiled transform" can
  continue to attach the exact pending transform after a compile-only step.
- Added a regression test that creates a pending compiled transform, simulates a correctness-failing
  score of that same transform, then verifies the repair call receives no autofill normalizer and
  the failed patch is reverted.

Verification:

- Focused tests:
  - `.venv/bin/python -m pytest tests/test_cli.py::test_repair_loop_does_not_autofill_pending_transform tests/test_cli.py::test_evolve_once_repairs_candidate_correctness_failure_before_finishing tests/test_cli.py::test_pending_transform_payload_normalizer_attaches_transform_to_score -q`: passed, 3 tests;
  - `.venv/bin/ruff check avo/cli.py tests/test_cli.py`: passed;
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 375 tests.

Decision:

- This is a repair-interface fix. A repair decision must now provide an explicit changed edit
  payload; the loop will not silently turn a repair retry into the same failed transform.

## 2026-05-10 - Checkpoint 5.04: Consume pending transforms scored during repair

Success criteria for this checkpoint:

- Prevent a compile-only transform from remaining "pending" after it is scored inside an immediate
  repair attempt.
- Preserve the existing behavior where a plain planning failure after compile-only still keeps the
  pending score follow-up alive.
- Verify attempt-history validation does not force a stale no-edit score after repair scoring.

Runtime fix:

- Attempt-history pending-transform detection now treats score payloads inside `repair_attempts` as
  real scores.
- This prevents the next top-level planning call from asking to score a transform that already
  failed correctness during the previous step's repair loop.
- Added a regression test for the exact shape: compile-only transform, repair-attempt score failure,
  then planning failure.

Verification:

- Focused tests:
  - `.venv/bin/python -m pytest tests/test_evolve.py::test_summarize_attempt_history_drops_compile_followup_after_repair_score tests/test_evolve.py::test_summarize_attempt_history_requests_score_after_compile_only_transform tests/test_evolve.py::test_summarize_attempt_history_keeps_compile_followup_after_planning_failure tests/test_cli.py::test_repair_loop_does_not_autofill_pending_transform -q`: passed, 4 tests;
  - `.venv/bin/ruff check avo/evolve.py tests/test_evolve.py avo/cli.py tests/test_cli.py`: passed;
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 376 tests.

Decision:

- The next live loop should not be nudged into rescoring the known-bad K async transform solely
  because its failed score was nested under repair attempts.

## 2026-05-10 - Checkpoint 5.05: Live loop after repair-memory fixes

Success criteria for this checkpoint:

- Verify the loop no longer treats the known-bad K async compile-only transform as pending after its
  failed score appeared under `repair_attempts`.
- Continue search from the Q-fragment-reuse best.
- Record CUDA evidence from the next bounded loop.

Research refresh:

- Exa again surfaced Dao-AILab's SM80 mainloop and NVIDIA CUTLASS's Ampere FA2 CuTe example.
- The useful signal was unchanged: real Ampere FA2 structure combines 128-bit `cp.async`, staged
  K/V, tensor-core MMA, register pipelines, and online-softmax/PV integration. Isolated shared
  staging remains a weak local move unless it changes consumed pipeline dataflow.

Run:

```bash
timeout 14400 .venv/bin/python -m avo evolve-loop \
  --lineage ./lineage \
  --knowledge knowledge/ampere.md \
  --cwd . \
  --env-file ../avo/.env.local \
  --timeout-s 3600 \
  --max-steps 4 \
  --compile-repair-attempts 3 \
  --attempts-dir ./attempts \
  --attempt-limit 40 \
  --loop-json attempts/loop_after_repair_memory_fix_20260510T0449Z.json
```

Result:

- Exit code `2`.
- `completed_steps=4`, `stopped_reason=max_steps`.
- No accepted candidate.
- The stale K async pending score did not repeat.

Step summary:

- Step 1 proposed K double-buffer shared-memory staging, but structural preflight rejected it because
  the new staging buffers were not connected to executable dataflow.
- Step 2 compiled a cooperative Q shared-memory staging transform:
  - ptxas reported 63 registers, 1 barrier, 14016 bytes shared memory, and no spills.
- Step 3 scored that exact Q staging transform:
  - `all_correct=true`;
  - geomean `7.645807821748137` TFLOPS;
  - rejected versus current best `8.960753680686471`.
- Step 4 failed planning validation after a repair attempt with ambiguous anchors and self-predicted
  correctness risk.

Decision:

- The repair-memory fixes worked in live state: the loop moved on instead of rescoring stale K async.
- Q shared-memory staging remains negative. Preserve Q in WMMA registers; do not spend more loop
  budget on standalone Q staging.

## 2026-05-10 - Checkpoint 5.06: Surface nested repair failures in attempt memory

Success criteria for this checkpoint:

- Make repair-loop failures visible to the next planner step instead of summarizing them only as
  `repairs=N`.
- Keep the summary compact enough to remain useful in long loops.
- Verify the live attempt summary exposes the ambiguous-anchor repair failure from checkpoint 5.05.

Runtime fix:

- Attempt-history summaries now include compact details for the last one or two repair attempts:
  failure class, command status, patch status, score status, and command.
- No-repair steps remain terse; they do not emit filler text.

Verification:

- Focused test:
  - `.venv/bin/python -m pytest tests/test_evolve.py::test_summarize_attempt_history_includes_repair_attempt_failures -q`: passed.
- Focused lint:
  - `.venv/bin/ruff check avo/evolve.py tests/test_evolve.py`: passed.
- Runtime summary sanity check:
  - latest summary now includes `repair_details=repair(class=structured_transform_preflight, ... expected exactly one anchor ...)`.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 377 tests.

Decision:

- This is an attempt-memory quality fix. It gives the planner the concrete nested repair failure
  without adding another CUDA-specific hard guard.

## 2026-05-10 - Checkpoint 5.07: K-staging follow-up after repair-summary fix

Success criteria for this checkpoint:

- Run a bounded loop after nested repair failures are visible in attempt memory.
- Score pending compile-only K cooperative staging evidence instead of leaving it as compile-only.
- Clean up any interrupted candidate patch before ending the checkpoint.

Runs:

```bash
timeout 14400 .venv/bin/python -m avo evolve-loop \
  --lineage ./lineage \
  --knowledge knowledge/ampere.md \
  --cwd . \
  --env-file ../avo/.env.local \
  --timeout-s 3600 \
  --max-steps 4 \
  --compile-repair-attempts 3 \
  --attempts-dir ./attempts \
  --attempt-limit 40 \
  --loop-json attempts/loop_after_repair_summary_fix_20260510T0501Z.json
```

```bash
timeout 7200 .venv/bin/python -m avo evolve-loop \
  --lineage ./lineage \
  --knowledge knowledge/ampere.md \
  --cwd . \
  --env-file ../avo/.env.local \
  --timeout-s 3600 \
  --max-steps 2 \
  --compile-repair-attempts 3 \
  --attempts-dir ./attempts \
  --attempt-limit 40 \
  --loop-json attempts/loop_score_pending_k_coop_20260510T0511Z.json
```

Result:

- First loop exit code `2`, `completed_steps=4`, `stopped_reason=max_steps`.
- Follow-up loop was interrupted during a later step after the useful score/repair chain had already
  been written; no loop JSON was emitted.
- No accepted candidate.

Evidence:

- K cooperative shared staging compiled with 64 registers, 1 barrier, 14016 bytes shared memory, and
  no spills.
- The matching full-target score passed correctness but regressed:
  - `all_correct=true`;
  - geomean `4.40940675249885` TFLOPS;
  - rejected versus current best `8.960753680686471`.
- A padded `k_tile[kTile][kHeadDim + 1]` variant compiled with 64 registers, 1 barrier, 14048 bytes
  shared memory, and no spills.
- Repair/scoring attempts for padded or transposed K shared-memory layouts failed correctness:
  - non-finite outputs;
  - CUDA unknown errors;
  - tolerance failure.

Cleanup:

- Interrupting the follow-up loop left an in-flight generated K-staging patch in the worktree.
- I reversed the current generated diff for `candidates/cuda_mma_attention/attention_kernel.cu` and
  verified the runtime repo worktree returned clean.
- No AVO compile/score/evolve process remained running.

Decision:

- Standalone K shared-memory staging is now strongly negative for this seed. It is either correct
  but far slower, or layout repairs fail correctness.
- Future K work should be part of a broader pipeline/dataflow change, not another isolated
  cooperative K staging variant.

## 2026-05-10 - Checkpoint 5.08: Add semantic-family memory for repeated CUDA moves

Success criteria for this checkpoint:

- Help the planner stop repeating the same broad CUDA optimization family after repeated
  unaccepted attempts.
- Keep the signal advisory instead of creating another hard ban or exact phrase filter.
- Classify the actual attempted edit, not historical rationale text, so old failure descriptions do
  not contaminate the family label.

Runtime fix:

- Attempt summaries now include a `family=...` label alongside the failure class.
- Recent unaccepted attempts are scanned for recurring transform families such as
  `shared_memory_staging`, `async_copy_pipeline`, `register_reuse`, WMMA contract/tile-shape
  changes, online-softmax/rescale changes, and scheduler/unroll changes.
- When one family repeats over the unaccepted tail, the summary emits a soft
  `Semantic-family signal` telling the planner to choose a materially different optimization
  family unless the next transform changes the dataflow, pipeline overlap, or validation contract.
- The classifier uses concrete edit payloads: `candidate_edit`, `candidate_transform`,
  `candidate_patch`, and the materialized patch. It deliberately avoids broad hypothesis/risk text
  because those often mention prior failed ideas.

Verification:

- Real attempt-history sanity check over the latest 12 attempts:
  - shared-memory staging attempts are labeled `family=shared_memory_staging`;
  - async-copy appears only where the actual edit or repair used async-copy machinery;
  - the emitted signal is `shared_memory_staging(count=11)`.
- Focused tests:
  - `.venv/bin/python -m pytest tests/test_evolve.py::test_summarize_attempt_history_flags_recurring_transform_family tests/test_evolve.py::test_summarize_attempt_history_flags_repeated_unaccepted_attempts tests/test_evolve.py::test_summarize_attempt_history_counts_mixed_recurring_failure_classes tests/test_evolve.py::test_summarize_attempt_history_includes_repair_attempt_failures -q`: passed, 4 tests.
- Lint and patch hygiene:
  - `.venv/bin/ruff check avo/evolve.py tests/test_evolve.py`: passed.
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 378 tests.

Decision:

- This addresses the root issue better than adding another guard: the planner now sees repeated
  failed semantic families and should move to a different coherent CUDA optimization move.
- The mechanism is still intentionally soft. It should steer search away from repeated standalone
  shared-memory staging without forbidding future shared-memory work that is part of a different
  dataflow or pipeline design.

## 2026-05-10 - Checkpoint 5.09: Relax async-copy granularity guidance

Success criteria for this checkpoint:

- Stop treating 16-byte async-copy granularity as a standalone hard rule.
- Keep real hard invariants in place: stage lifecycle, initialized shared state, address ownership,
  and disconnected helper/skeleton edits.
- Preserve the performance guidance that 16-byte groups are usually the useful sm86 throughput
  target for BF16 `cp.async`.

Runtime/knowledge clarification:

- Runtime structural preflight already allows scalar BF16 and non-scalar
  `__pipeline_memcpy_async` size expressions through compile/repair.
- I softened the remaining hard wording in `knowledge/ampere.md` and
  `knowledge/retrieval_claims.md`:
  - 16-byte groups are preferred performance guidance, not a hard rejection by themselves;
  - narrower copies are acceptable for coherent API probes or disjoint tail handling;
  - structural rejection should stay focused on lifecycle, initialized-data, address-ownership, and
    no-effect/skeleton failures.

Verification:

- Remaining hard-phrasing scan now shows only advisory wording such as `copy granularity alone is
  not a hard rejection`.
- Targeted tests:
  - `.venv/bin/python -m pytest tests/test_knowledge.py tests/test_agent.py::test_structural_preflight_allows_scalar_bf16_async_copy_for_repair tests/test_agent.py::test_structural_preflight_allows_non_scalar_async_copy_size_expression tests/test_agent.py::test_decision_feedback_explains_scalar_bf16_async_copy_error -q`: passed, 44 tests.
- Patch hygiene:
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 378 tests.

Decision:

- Keep async-copy granularity as guidance the agent can reason about, not a gate that blocks a
  coherent transform before the compiler/repair loop can see it.

## 2026-05-10 - Checkpoint 5.10: Longer loop accepts probability-fragment reuse

Success criteria for this checkpoint:

- Run the evolve loop after semantic-family memory and async-copy guidance changes.
- Verify whether the loop now chooses materially different CUDA moves instead of repeating isolated
  shared-memory staging.
- Confirm any accepted candidate with correctness, TFLOPS, compile-resource evidence, and tests.

Run:

```bash
timeout 21600 .venv/bin/python -m avo evolve-loop \
  --lineage ./lineage \
  --knowledge knowledge/ampere.md \
  --cwd . \
  --env-file ../avo/.env.local \
  --timeout-s 5400 \
  --max-steps 8 \
  --compile-repair-attempts 4 \
  --attempts-dir ./attempts \
  --attempt-limit 48 \
  --loop-json attempts/loop_after_semantic_family_async_softening_20260510T0540Z.json
```

Result:

- Loop stopped early with `accepted=true`, `completed_steps=6`, `stopped_reason=accepted`.
- The lineage repo committed the accepted candidate as `e96ce95 evolve: accept candidate`.
- The first step still failed planning validation by repeating an unpatched MMA seed score.
- A `kThreads=256` work-mapping transform compiled, then scored correct but regressed to
  `5.788182135598767` geomean TFLOPS versus best `8.960753680686471`.
- A later `kTile=32` proposal was rejected by shape-contract preflight because it did not implement
  the necessary second 16-row sub-tile dataflow.

Accepted transform:

- Actual materialized CUDA change: hoist the PV-side `probability_frag` declaration and
  `wmma::load_matrix_sync(probability_frag, probabilities, kTile)` out of the 8-output-chunk loop.
- This loads the probability WMMA A fragment once per key tile instead of once per output chunk.
- The planner's first sentence overstated the move as V-tile reuse. The patch does not reduce V
  global loads; V is still loaded once per output chunk.

Gate score:

- All 8 full-target BF16 cases correct.
- Accepted geomean: `9.157629176515384` TFLOPS versus previous best
  `8.960753680686471`.
- Per-case TFLOPS:
  - seq4096: noncausal `12.93389270271075`, causal `6.61122454205263`;
  - seq8192: noncausal `12.897203926337777`, causal `6.575085128380674`;
  - seq16384: noncausal `12.806310533173857`, causal `6.5326744387412985`;
  - seq32768: noncausal `12.819775713671431`, causal `6.3600546666917275`.

Confirmation:

- Re-scored the accepted candidate with repeats 3 and warmup 2:
  - `all_correct=true`;
  - geomean `9.237725222665237` TFLOPS.
- Standalone compile:
  - sm86 target;
  - 64 registers;
  - 1 barrier;
  - 9920 bytes shared memory;
  - 0 spill stores/loads.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 378 tests.

Decision:

- Preserve both accepted reuse moves: `q_frags[8]` across K tiles and `probability_frag` across PV
  output chunks.
- This is evidence that the semantic-family signal helped the loop escape repeated staging attempts,
  but planner-interface reliability is still imperfect: it still produced one unpatched-score
  validation failure and one invalid batch-size failure in the same run.
- Future V/PV proposals must distinguish actual V reuse from probability-fragment reuse; do not
  accept rationale text as evidence of what changed.

## 2026-05-10 - Checkpoint 5.11: Sharpen planner retry feedback for loop failures

Success criteria for this checkpoint:

- Use the failures observed in checkpoint 5.10 to improve the retry loop without adding another
  CUDA-specific guard.
- Make the correction for repeated unpatched MMA scores unambiguous: no no-edit score, no raw CUDA
  patch, use structured-transform mode.
- Make the correction for invalid transform batch sizes explicit before the attempt is logged.

Runtime fix:

- The validation feedback for recorded unpatched MMA seed scores now says to set
  `edit_mode='transform'`, set `candidate_patch` to `''`, and provide one exact
  `candidate_transform` operation or a scoped wrapper/kernel batch.
- A new feedback hint handles `candidate_transform batch steps must contain 1 to 8 operations`:
  return 1 to 8 primitive operations, use a single non-batch operation, or split oversized batches
  into the smallest coherent semantic move.

Verification:

- Focused tests:
  - `.venv/bin/python -m pytest tests/test_agent.py::test_decision_feedback_explains_unpatched_mma_score_error tests/test_agent.py::test_decision_feedback_explains_transform_batch_size_error -q`: passed, 2 tests.
- Lint and patch hygiene:
  - `.venv/bin/ruff check avo/agent.py tests/test_agent.py`: passed.
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 379 tests.

Decision:

- This is an agent-interface repair, not a new CUDA ban. It should help the planner fix malformed
  decision payloads before the evolve loop wastes a step recording a planning-validation failure.

## 2026-05-10 - Checkpoint 5.12: Validate claimed operand load-reuse semantics

Success criteria for this checkpoint:

- Address the mismatch seen in checkpoint 5.10 where the planner claimed V-tile reuse but the
  materialized patch only hoisted `probability_frag`.
- Keep the check structural and semantic, not another historical phrase ban.
- Allow the accepted probability-fragment hoist when the claim accurately describes it.

Runtime fix:

- Candidate-transform validation now inspects planning text for concrete operand load-reduction or
  reuse claims, such as reducing/reusing Q/K/V/probability loads.
- For `replace_once` transforms, the validator compares before/after operand load sites:
  - it accepts a claim if the replacement reduces the operand load-site count;
  - it also accepts a hoist if the operand load moves from inside a repeated loop to outside it.
- It rejects claims where the named operand's load sites are unchanged and still inside the repeated
  loop.

Verification:

- Regression tests:
  - claimed V reuse with only `probability_frag` hoisting is rejected;
  - the same transform is accepted when the claim says probability load hoist;
  - existing contract-only semantic mismatch test still passes.
- Focused command:
  - `.venv/bin/python -m pytest tests/test_agent.py::test_parse_variation_decision_rejects_claimed_v_reuse_without_v_load_change tests/test_agent.py::test_parse_variation_decision_allows_probability_load_hoist_claim tests/test_agent.py::test_parse_variation_decision_rejects_constant_proxy_for_dataflow_claim -q`: passed, 3 tests.
- Lint and patch hygiene:
  - `.venv/bin/ruff check avo/agent.py tests/test_agent.py`: passed.
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 381 tests.

Decision:

- This should make the agent more honest about the actual CUDA move it materializes. It does not
  forbid V reuse; it requires a V-reuse claim to actually move or reduce V load sites.

## 2026-05-10 - Checkpoint 5.13: Follow-up loop accepts kThreads=64 retune

Success criteria for this checkpoint:

- Run the evolve loop after claimed-load-reuse validation.
- Check whether the loop distinguishes real V reuse from probability-fragment reuse.
- Confirm any accepted candidate with a higher-repeat score before committing it to the runtime repo.

Run:

```bash
timeout 21600 .venv/bin/python -m avo evolve-loop \
  --lineage ./lineage \
  --knowledge knowledge/ampere.md \
  --cwd . \
  --env-file ../avo/.env.local \
  --timeout-s 5400 \
  --max-steps 8 \
  --compile-repair-attempts 4 \
  --attempts-dir ./attempts \
  --attempt-limit 56 \
  --loop-json attempts/loop_after_load_reuse_semantic_validation_20260510T0615Z.json
```

Result:

- Loop stopped with `accepted=true`, `completed_steps=7`, `stopped_reason=accepted`.
- The lineage repo committed `8326b53 evolve: accept candidate`.
- The first step still failed planning validation because self-invalid detection treated historical
  discussion of a previous bad K-staging repair as if the current proposal admitted it was invalid.
- A cooperative K shared-memory staging candidate compiled and scored correct, but regressed to
  `4.393552873015184` geomean TFLOPS versus best `9.157629176515384`.
- A true V-fragment register-cache candidate compiled with 110 registers and scored correct, but
  regressed to `8.359720357318682` geomean TFLOPS.
- The accepted candidate changed `kThreads` from 128 to 64.

Accepted candidate:

- Gate score:
  - all 8 full-target BF16 cases correct;
  - geomean `9.168741394385114` TFLOPS versus previous gate best
    `9.157629176515384`.
- Confirmation score with repeats 3 and warmup 2:
  - all 8 full-target BF16 cases correct;
  - geomean `9.254126656665425` TFLOPS.
- Compile evidence from the loop:
  - sm86 target;
  - 64 registers;
  - 1 barrier;
  - 9920 bytes shared memory;
  - 0 spill stores/loads.

Verification:

- Full runtime suite after applying the accepted candidate:
  - `.venv/bin/python -m pytest -q`: passed, 381 tests.

Decision:

- Keep `kThreads=64` as the current accepted runtime state, but treat it as a small resource retune,
  not a major dataflow improvement.
- Preserve the two dataflow wins already accepted: `q_frags[8]` reuse and `probability_frag` hoist.
- Next reliability fix should narrow the self-invalid detector so historical discussion of a prior
  failure does not invalidate a current corrected transform.

## 2026-05-10 - Checkpoint 5.14: Historical failure notes no longer self-invalidate current transforms

Success criteria for this checkpoint:

- Fix the planning validation bug observed in the 06:15Z evolve loop where a note about a previous
  failed K-staging transform invalidated the current candidate.
- Keep current self-invalid candidates rejected when the decision says the patch will fail or should
  not be scored.
- Verify the change with focused and full tests.

Change:

- Runtime `avo/agent.py` now treats planning windows that start as historical prior-failure context
  and contain a failure signal as evidence, not as a current self-rejection.
- The same historical-context filter is used by contract-only semantic validation, so a
  `set_constexpr_int` retune is not rejected merely because the planner mentions a prior failed
  K/shared-memory staging attempt.
- Added a regression test using the concrete failure shape from the loop: a valid `kThreads=64`
  transform whose risk text says the previous K tile shared-memory staging transform failed with
  `key_start` undefined.

Verification:

- Focused regression and existing self-rejection tests:
  - `.venv/bin/python -m pytest tests/test_agent.py::test_parse_variation_decision_allows_historical_failure_note tests/test_agent.py::test_parse_variation_decision_rejects_self_rejected_patch tests/test_agent.py::test_parse_variation_decision_rejects_do_not_use_this_diff_warning tests/test_agent.py::test_parse_variation_decision_rejects_do_not_score_self_invalid_patch -q`: passed, 4 tests.
- Agent tests:
  - `.venv/bin/python -m pytest tests/test_agent.py -q`: passed, 189 tests.
- Patch hygiene:
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 382 tests.

Decision:

- This is a small reliability fix, not a new preflight class. It reduces the reactive guard behavior
  by allowing the planner to use prior failure information while still rejecting decisions that call
  their own current edit invalid.

## 2026-05-10 - Checkpoint 5.15: Longer loop records work-mapping negatives and narrows load-claim validation

Success criteria for this checkpoint:

- Run a longer evolve loop after the historical-failure-note fix.
- Check whether the planner can carry compile-only transforms into score attempts.
- Record useful CUDA negative evidence from the run.
- Fix any recurring validation issue that blocks reasonable planning context.

Run:

```bash
timeout 28800 .venv/bin/python -m avo evolve-loop \
  --lineage ./lineage \
  --knowledge knowledge/ampere.md \
  --cwd . \
  --env-file ../avo/.env.local \
  --timeout-s 7200 \
  --max-steps 12 \
  --compile-repair-attempts 4 \
  --attempts-dir ./attempts \
  --attempt-limit 72 \
  --loop-json attempts/loop_after_historical_failure_fix_20260510T0628Z.json
```

Loop result:

- Stopped with `accepted=false`, `completed_steps=12`, `stopped_reason=max_steps`.
- The pending-transform feedback path did work for compile-only candidates:
  - `kThreads=32` lane-index rewrite compiled, was then scored, and failed correctness.
  - `kThreads=128` compiled, was then scored, passed correctness, and regressed to
    `9.118922525821796` geomean TFLOPS versus best `9.168741394385114`.
  - pure `kThreads=32` compiled, was then scored, passed correctness, and regressed to
    `8.357124079366539`.
  - `kQueryTilesPerBlock=2` compiled, was then scored, passed correctness, and regressed to
    `9.111694101686032`.
- The run exposed two reliability issues:
  - after the failed 32-thread lane-index score, the planner proposed a revert transform even though
    cleanup had already restored the source, producing a no-op transform rejection;
  - load-reuse semantic validation treated baseline context like "the current kernel already loads Q
    once" as if the proposed transform claimed to newly reduce Q loads.

Fix:

- Runtime `avo/agent.py` now skips existing-state planning windows when looking for load-reduction
  claims or contract-only dataflow claims.
- Added tests that allow baseline context such as "the current kernel already loads Q once", while
  still rejecting "this transform loads Q once" when the structured transform does not change Q load
  sites.
- Runtime knowledge now records the thread-count and multi-query-tile negative evidence.

Verification:

- Focused load-claim and historical-context tests:
  - `.venv/bin/python -m pytest tests/test_agent.py::test_parse_variation_decision_allows_existing_q_reuse_context tests/test_agent.py::test_parse_variation_decision_rejects_current_q_reuse_claim_without_q_load_change tests/test_agent.py::test_parse_variation_decision_rejects_claimed_v_reuse_without_v_load_change tests/test_agent.py::test_parse_variation_decision_allows_probability_load_hoist_claim tests/test_agent.py::test_parse_variation_decision_allows_historical_failure_note -q`: passed, 5 tests.
- Agent tests:
  - `.venv/bin/python -m pytest tests/test_agent.py -q`: passed, 191 tests.
- Lint and patch hygiene:
  - `.venv/bin/ruff check avo/agent.py tests/test_agent.py`: passed.
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 384 tests.

Decision:

- Keep `kThreads=64` as the current accepted thread-count point.
- Treat simple `kThreads=32`, `kThreads=128`, and serial `kQueryTilesPerBlock=2` as negative evidence
  for this seed.
- The no-op revert-after-cleanup behavior is still a follow-up reliability issue; the loop should
  classify a failed scored family and move on rather than trying to revert an already-clean source.

## 2026-05-10 - Checkpoint 5.16: Correctness repair prompt rejects revert-only thinking

Success criteria for this checkpoint:

- Address the no-op revert behavior observed after the failed 32-thread lane-index score.
- Keep the change as planner feedback rather than a brittle phrase ban.
- Verify the repair prompt and full runtime suite.

Change:

- Runtime `avo/cli.py` correctness-repair feedback now says cleanup has already restored the clean
  pre-edit source.
- It explicitly tells the planner not to return a revert-only repair because the repair payload must
  make a real source change against the current source.
- If the failed family should be abandoned, the prompt directs the planner to choose a different
  coherent transform family instead of restoring baseline code.

Verification:

- Focused CLI correctness-repair test:
  - `.venv/bin/python -m pytest tests/test_cli.py::test_evolve_once_repairs_candidate_correctness_failure_before_finishing -q`: passed.
- Lint:
  - `.venv/bin/ruff check avo/cli.py tests/test_cli.py`: passed.
- Grouped runtime tests:
  - `.venv/bin/python -m pytest tests/test_cli.py tests/test_agent.py tests/test_knowledge.py -q`: passed, 267 tests.
- Patch hygiene:
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 384 tests.

Decision:

- This keeps correctness repair available, but makes the interface clearer: repair means a revised
  executable semantic move on clean source, not undoing a patch the loop already cleaned up.

## 2026-05-10 - Checkpoint 5.17: Add semantic families for work-mapping retries

Success criteria for this checkpoint:

- Improve the search loop's memory for recurring failed optimization families without adding a hard
  phrase ban.
- Cover the repeated families seen in the 06:28Z loop: thread-count retunes and query-tile
  work-mapping changes.
- Verify that attempt-history summaries now surface those families.

Research/context:

- Exa search surfaced Reflexion-style agent papers that treat environment feedback as a semantic
  memory signal, not only as immediate retry text. The useful design point for AVO is to generalize
  failures into compact family labels that steer future attempts.
- The local `pi-mono-agent` package shows a similar interface pattern at a lower level: validated
  tool preflight and explicit tool-result events are first-class loop boundaries. For AVO, the
  equivalent boundary is the structured candidate transform plus scored attempt summary.

Change:

- Runtime `avo/evolve.py` now classifies `kThreads`/`kWarps` retunes and thread/warp mapping text as
  `thread_count_or_warp_mapping`.
- Runtime `avo/evolve.py` now classifies `kQueryTilesPerBlock` and query-tiles-per-block rewrites as
  `query_tile_work_mapping`.
- Added regression tests showing that three unaccepted attempts in either family trigger the existing
  semantic-family signal.

Verification:

- Focused evolve tests:
  - `.venv/bin/python -m pytest tests/test_evolve.py::test_summarize_attempt_history_flags_thread_count_family tests/test_evolve.py::test_summarize_attempt_history_flags_query_tile_work_mapping_family tests/test_evolve.py::test_summarize_attempt_history_flags_recurring_transform_family -q`: passed, 3 tests.
- Evolve tests:
  - `.venv/bin/python -m pytest tests/test_evolve.py -q`: passed, 80 tests.
- Lint and patch hygiene:
  - `.venv/bin/ruff check avo/evolve.py tests/test_evolve.py`: passed.
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 386 tests.
- Real attempt-history check:
  - `summarize_attempt_history(Path("attempts"), limit=12)` now classifies the 06:28Z loop's
    `kThreads=32`, `kThreads=128`, and scored thread retunes as
    `thread_count_or_warp_mapping`.
  - The same summary classifies the `kQueryTilesPerBlock=2` compile and score attempts as
    `query_tile_work_mapping`.
  - The semantic-family signal now reports `thread_count_or_warp_mapping(count=6)` on the real
    unaccepted tail.

Decision:

- This makes the loop more likely to move away from exhausted work-mapping families after repeated
  measured regressions, while still allowing a future thread/work-mapping candidate if it changes the
  actual dataflow, synchronization, or validation contract.

## 2026-05-10 - Checkpoint 5.18: Accept kThreads=96 after family-guided loop

Success criteria for this checkpoint:

- Run the evolve loop after adding semantic-family feedback for repeated work-mapping attempts.
- Confirm whether the planner can carry a compiled candidate forward to scoring.
- Accept only a correctness-passing target-workload improvement.

Result:

- Loop command:
  - `timeout 28800 .venv/bin/python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --timeout-s 7200 --max-steps 10 --compile-repair-attempts 4 --attempts-dir ./attempts --attempt-limit 80 --loop-json attempts/loop_after_work_mapping_family_signal_20260510T0707Z.json`
- The loop stopped after 8 completed steps with `accepted=true`.
- Accepted candidate:
  - `candidates/cuda_mma_attention/attention_kernel.cu`: `constexpr int kThreads = 64;` to
    `constexpr int kThreads = 96;`
- Gate:
  - Previous best geomean: `9.168741394385114` TFLOPS.
  - Candidate geomean: `9.451443582484515` TFLOPS.
  - Correctness: passed all target large-shape BF16 cases scored by the loop.

Observed loop behavior:

- The planner did move beyond the exact earlier `kThreads=32/128` and query-tile attempts, but it
  still produced some malformed or self-invalid structured transforms before reaching the accepted
  candidate.
- The loop successfully preserved a compile-only candidate across the follow-up scoring step instead
  of losing the pending transform.
- The accepted improvement is small but valid: a scoped occupancy retune on the realistic target
  workload, not a tiny-shape artifact.

Decision:

- Keep the accepted `kThreads=96` retune as the new runtime candidate.
- The next loop-mechanics fix should focus on historical failure context and self-invalid risk text:
  the planner can still mention prior invalid attempts in a way that trips current-attempt validation.

## 2026-05-10 - Checkpoint 5.19: Soften async-copy granularity and fix framed failure notes

Success criteria for this checkpoint:

- Do not turn async-copy copy width into a standalone hard rejection. Keep it as performance
  guidance unless the candidate violates a structural invariant.
- Allow the planner to cite framed historical failures such as "the repair request shows the
  previous attempt failed" without treating that as an admission that the current candidate is bad.
- Still reject a current/proposed candidate that says it will be rejected by structural preflight.

Research/context:

- NVIDIA's CUDA Programming Guide says LDGSTS async copies support 4, 8, or 16 byte transfers, with
  16-byte copies enabling L1-bypass behavior and better performance conditions.
- NVIDIA's Ampere Tuning Guide frames async copy as a global-to-shared feature for overlapping
  compute with data movement and avoiding extra registers. That supports treating copy width as a
  performance contract inside a coherent pipeline, not as a reason to block every scalar repair.

Change:

- Runtime `avo/agent.py` now recognizes framed historical failure context even when the sentence
  starts with evidence framing such as "the immediate compile repair request shows...".
- Runtime `avo/agent.py` now rejects current/proposed transforms that explicitly say they will be
  rejected by structural preflight or validation.
- Runtime prompt guidance now says async-copy vector width is preferred, but copy-width preference
  alone is not a hard rejection when a transform is coherent and repairable.

Verification:

- Focused planning/async tests:
  - `.venv/bin/python -m pytest tests/test_agent.py::test_parse_variation_decision_allows_historical_failure_note tests/test_agent.py::test_parse_variation_decision_allows_framed_historical_failure_note tests/test_agent.py::test_parse_variation_decision_rejects_current_transform_preflight_failure tests/test_agent.py::test_structural_preflight_allows_scalar_bf16_async_copy_for_repair tests/test_agent.py::test_structural_preflight_allows_non_scalar_async_copy_size_expression tests/test_agent.py::test_parse_variation_decision_rejects_invalid_async_pipeline_lifecycle -q`: passed, 6 tests.
- Agent tests:
  - `.venv/bin/python -m pytest tests/test_agent.py -q`: passed, 193 tests.
- Lint and patch hygiene:
  - `.venv/bin/ruff check avo/agent.py tests/test_agent.py`: passed.
  - `git diff --check`: passed.

Decision:

- This keeps hard preflight on structural CUDA mistakes: invalid pipeline API shape, invalid stage
  lifecycle, unsupported WMMA contracts, disconnected helpers, and stale symbols.
- It avoids blocking repairable async-copy experiments solely because they used scalar BF16 copy
  size while the agent is still developing a coherent pipeline.

## 2026-05-10 - Checkpoint 5.20: Accept score-store syncwarp refinement

Success criteria for this checkpoint:

- Validate the async-copy/preflight softening by running a longer evolve loop.
- Keep the loop on realistic target shapes.
- Accept only a correctness-passing throughput improvement.
- Record both accepted and rejected transform families as search evidence.

Loop command:

- `timeout 36000 .venv/bin/python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --timeout-s 9600 --max-steps 16 --compile-repair-attempts 4 --attempts-dir ./attempts --attempt-limit 96 --loop-json attempts/loop_after_async_preflight_softening_20260510T0804Z.json`

Result:

- The loop stopped after 12 completed steps with `accepted=true`.
- Accepted candidate:
  - `candidates/cuda_mma_attention/attention_kernel.cu`: add `__syncwarp();` immediately before
    the existing `__syncthreads();` after `wmma::store_matrix_sync(scores, ...)`.
- Gate:
  - Previous best geomean: `9.451443582484515` TFLOPS.
  - Candidate geomean: `9.507832270603132` TFLOPS.
  - Correctness: passed all 8 target large-shape BF16 cases.

Useful negative evidence:

- Cooperative K/V shared staging with padding passed correctness but regressed to
  `3.640496322360991` TFLOPS.
- `kThreads=128` passed correctness but regressed to `9.16374468686178` TFLOPS versus the current
  `kThreads=96` candidate.
- Single-stage K async-copy staging compiled, then failed target scoring with CUDA launch/unknown
  errors and geomean `0.0`.
- K-only shared staging passed correctness but regressed to `3.6932886091263213` TFLOPS.

Loop behavior:

- The planner recovered from an invalid request for profiler metrics through `avo score`.
- The loop carried compile-only candidates forward to scoring for K/V staging, `kThreads=128`,
  async K staging, K-only staging, and the accepted syncwarp transform.
- The new self-invalid risk filter rejected a candidate whose own risk text predicted incorrect
  stage indexing before wasting a compile.

Decision:

- Keep `kThreads=96` and the score-store `__syncwarp()` refinement as the current runtime state.
- Treat plain synchronous K/KV staging as exhausted negative evidence for this seed.
- Future async-copy work should focus on correct stage lifecycle and initialized-data guarantees,
  not copy width alone.

Follow-up confirmation:

- NVIDIA synchronization guidance supports explicit warp synchronization for warp-synchronous
  shared/global-memory exchange on modern independent-thread-scheduling GPUs.
- Confirmation score:
  - `.venv/bin/python -m avo score --backend candidate --candidate candidates/cuda_mma_attention_seed.py --seq-lens 4096,8192,16384,32768 --total-tokens 32768 --num-heads 16 --head-dim 128 --dtype bf16 --causal both --repeats 3 --warmup 2 --timeout-s 600`
  - Result: all 8 target cases correct.
  - Geomean: `9.576586797806204` TFLOPS.

Decision:

- The syncwarp candidate is no longer only a one-sample gate win; keep it as confirmed current
  runtime state.

## 2026-05-10 - Checkpoint 5.21: Fix transform-family classifier priority

Success criteria for this checkpoint:

- Ensure semantic-family memory reflects the actual transform family, not incidental CUDA context.
- Prevent cooperative `blockDim.x` loops from being mislabeled as thread-count/work-mapping edits.
- Keep exact `kThreads`/`kWarps` retunes classified as thread-count work even when diff context
  contains shared-memory declarations.
- Classify accepted `__syncwarp` edits as synchronization/barrier work.

Issue:

- The real attempt summary after Checkpoint 5.20 labeled K/KV shared-staging, async-copy, and
  syncwarp attempts as `thread_count_or_warp_mapping` because the classifier treated `blockDim.x`
  and noisy diff context as thread-work signals.

Change:

- Runtime `avo/evolve.py` now prioritizes exact `kThreads`/`kWarps` constexpr retunes first.
- Runtime `avo/evolve.py` now classifies async-copy, `__syncwarp`, shared-memory staging, and
  generic barriers before broad thread-work text.
- Removed bare `blockDim.x` as a thread-count family trigger.
- Added regression tests for shared staging with `blockDim.x`, syncwarp edits, and noisy shared
  context around `kThreads` retunes.

Verification:

- Focused tests:
  - `.venv/bin/python -m pytest tests/test_evolve.py::test_summarize_attempt_history_keeps_blockdim_shared_staging_family tests/test_evolve.py::test_summarize_attempt_history_classifies_syncwarp_family tests/test_evolve.py::test_summarize_attempt_history_flags_thread_count_family tests/test_evolve.py::test_summarize_attempt_history_flags_recurring_transform_family -q`: passed, 4 tests.
- Evolve tests:
  - `.venv/bin/python -m pytest tests/test_evolve.py -q`: passed, 82 tests.
- Lint and patch hygiene:
  - `.venv/bin/ruff check avo/evolve.py tests/test_evolve.py`: passed.
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 390 tests.
- Real attempt-history check:
  - K/KV staging attempts now show `family=shared_memory_staging`.
  - Async K staging now shows `family=async_copy_pipeline`.
  - `kThreads=96`/`kThreads=128` retunes now show `family=thread_count_or_warp_mapping`.
  - The accepted syncwarp compile/score now shows `family=synchronization_or_barrier`.

Decision:

- This improves the search loop's memory without adding a new ban. Future planner guidance should
  now steer away from exhausted shared-staging families without misclassifying unrelated CUDA edits.

## 2026-05-10 - Checkpoint 5.22: Reject noisy syncwarp-removal acceptance

Success criteria for this checkpoint:

- Verify whether the classifier-guided follow-up loop picks a different semantic family after
  repeated shared-staging regressions.
- Do not keep a loop-accepted kernel edit unless it survives confirmation beyond one-shot gate
  noise.

Loop:

- Command:
  - `timeout 36000 .venv/bin/python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --timeout-s 9600 --max-steps 12 --compile-repair-attempts 4 --attempts-dir ./attempts --attempt-limit 112 --loop-json attempts/loop_after_family_classifier_fix_20260510T1118Z.json`
- Result:
  - `accepted=true`
  - `completed_steps=6`
  - `stopped_reason=accepted`

Evidence:

- The first attempt failed semantic validation because it claimed reduced/reused Q loads without a
  transform that actually changed Q load sites.
- A single-stage K shared-memory staging transform compiled and passed correctness, then regressed
  to `3.594328062594656` geomean TFLOPS versus best `9.507832270603132`. This is correctly
  `shared_memory_staging` negative evidence, not thread-count evidence.
- Two planner turns were invalid orchestration decisions:
  - one tried to repeat a successful compile-only transform instead of scoring it or changing
    family;
  - one requested `profile`, which is unavailable in this runtime.
- The final candidate removed the explicit `__syncwarp()` after
  `wmma::store_matrix_sync(scores, ...)`, leaving the following `__syncthreads()`. The one-shot
  loop gate accepted it at `9.53940653329568` geomean TFLOPS versus gate best
  `9.507832270603132`.

Confirmation:

- Removed-syncwarp candidate:
  - `.venv/bin/python -m avo score --backend candidate --candidate candidates/cuda_mma_attention_seed.py --seq-lens 4096,8192,16384,32768 --total-tokens 32768 --num-heads 16 --head-dim 128 --dtype bf16 --causal both --repeats 3 --warmup 2 --timeout-s 600`
  - Result: all 8 target cases correct.
  - Geomean: `9.537755900752106` TFLOPS.
- Restored-syncwarp A/B under the same confirmation settings:
  - Result: all 8 target cases correct.
  - Geomean: `9.544394274937641` TFLOPS.
- Previous syncwarp confirmation remains stronger at `9.576586797806204` geomean TFLOPS.

Decision:

- Do not keep or commit the syncwarp-removal candidate. The worktree is back to the committed
  syncwarp-present kernel.
- Treat the loop acceptance as measurement noise around a synchronization boundary, not as a real
  kernel improvement.
- The classifier fix helped: the loop did score another shared-staging variant, recorded it as a
  shared-memory regression, and then moved to a synchronization-family candidate. The remaining
  weakness is command selection and acceptance confidence, not raw CUDA diff handling.

## 2026-05-10 - Checkpoint 5.23: Add one-shot acceptance confidence and command recovery

Success criteria for this checkpoint:

- Prevent near-tie one-shot timing scores from entering lineage as accepted improvements.
- Stop advertising `avo profile` to the planner when the runtime cannot support Nsight/CUPTI.
- Recover from a repeated successful compile-only transform by scoring the pending transform when
  the target seed can be inferred.
- Preserve the corrected local lineage state after the noisy syncwarp-removal acceptance.

Change:

- Runtime `avo/lineage.py` now rejects one-shot score payloads (`repeats<=1`, `warmup<=1`) unless
  they improve the current same-signature best by at least `0.5%`. Low-margin candidates must be
  rerun with `repeats>=3` and `warmup>=2` or find a larger improvement.
- Runtime `avo/agent.py` now builds the variation prompt from runtime profiler availability:
  when the Thunder/CUPTI marker is present, the allowed command list is `env, compile, score` and
  profile is described as unavailable.
- Runtime `avo/cli.py` now normalizes a planner payload that repeats the exact pending successful
  compile-only transform into a target `avo score` command for the matching seed family when the
  transform path/source identifies MMA, warp-rows, or tiled attention.
- Reverted the ignored nested `lineage/` commit that had recorded the noisy syncwarp-removal
  candidate, leaving lineage latest at the syncwarp-present score
  `9.507832270603132`.

Verification:

- Focused tests:
  - `.venv/bin/python -m pytest tests/test_lineage.py tests/test_cli.py::test_pending_transform_payload_normalizer_attaches_transform_to_score tests/test_cli.py::test_pending_transform_payload_normalizer_rewrites_repeated_mma_compile_to_score tests/test_agent.py::test_build_variation_prompt_excludes_profile_when_unavailable tests/test_agent.py::test_decision_feedback_explains_unavailable_profile_error -q`: passed, 24 tests.
- Affected suites:
  - `.venv/bin/python -m pytest tests/test_lineage.py tests/test_cli.py tests/test_agent.py -q`: passed, 251 tests.
- Lint and patch hygiene:
  - `.venv/bin/ruff check avo/lineage.py avo/agent.py avo/cli.py tests/test_lineage.py tests/test_cli.py tests/test_agent.py`: passed.
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 396 tests.
- Real noisy-candidate regression check:
  - New gate decision for the removed-syncwarp loop score versus corrected local lineage best:
    `accepted=false`, reason `candidate improvement is within one-shot timing noise; rerun with
    repeats>=3/warmup>=2 or improve geomean by at least 0.5%`.

Decision:

- Keep the syncwarp-present runtime kernel and corrected local lineage.
- Future small timing wins can still be accepted, but only after stronger score settings or a
  margin large enough to survive the one-shot noise filter.

## 2026-05-10 - Checkpoint 5.24: Mark stale accepted attempt history

Success criteria for this checkpoint:

- Verify the one-shot acceptance gate and command recovery in a fresh evolve loop.
- Prevent reverted or noisy accepted attempt records from polluting future planner context as if
  they were the current lineage best.

Loop:

- Command:
  - `timeout 36000 .venv/bin/python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --timeout-s 9600 --max-steps 12 --compile-repair-attempts 4 --attempts-dir ./attempts --attempt-limit 128 --loop-json attempts/loop_after_one_shot_gate_20260510T1142Z.json`
- Result:
  - `accepted=false`
  - `completed_steps=12`
  - `stopped_reason=max_steps`

Evidence:

- Self-repair worked on a vectorized K-staging compile failure; the repaired transform compiled
  and was then scored instead of being lost.
- Pending-transform command recovery worked for the repaired K-staging candidate: the loop scored
  the compiled pending transform and rejected it after a correct but slow
  `3.626402277701616` geomean TFLOPS run.
- A second cooperative K-staging transform also passed correctness but regressed to
  `3.614179233547189` geomean TFLOPS, reinforcing that isolated K shared staging remains exhausted
  for this seed.
- Several smaller semantic moves passed correctness but regressed:
  - removing a barrier after score storage scored `9.449127781532056`;
  - `kThreads=64` scored `9.100221174050498`;
  - V-loop unroll scored `9.419139786612444`;
  - `kThreads=80` scored `9.002945885589778`.
- A `kTile=8` proposal was rejected before compile as no source change.
- A key-loop unroll and a V-fragment pipeline transform compiled near the end of the loop but were
  not scored before the step budget expired.
- The planner still repeatedly described the noisy removed-syncwarp attempt as the current
  `9.54` TFLOPS winner because the raw attempt-history directory contained a historical
  `gate accepted=true` record even though local lineage had been reverted to the syncwarp-present
  `9.507832270603132` best.

Change:

- Runtime attempt-history summarization now receives the actual lineage best from
  `best_geomean(args.lineage)`.
- Accepted score records above the current lineage best are labeled `class=stale_accepted` and
  annotated as reverted or noisy historical acceptances, not current best state.
- The summary also adds a lineage-correction note when recent attempt history contains such stale
  accepted records.

Verification:

- Focused tests:
  - `.venv/bin/python -m pytest tests/test_evolve.py::test_summarize_attempt_history_marks_reverted_acceptance_stale tests/test_evolve.py::test_summarize_attempt_history_reports_recent_steps -q`: passed, 2 tests.
- Affected suites:
  - `.venv/bin/python -m pytest tests/test_evolve.py tests/test_cli.py -q`: passed, 119 tests.
- Lint and patch hygiene:
  - `.venv/bin/ruff check avo/evolve.py avo/cli.py tests/test_evolve.py`: passed.
  - `git diff --check`: passed.
- Real attempt-history check:
  - `attempts/2026-05-10T11-26-29-00-00.json` is now summarized as
    `class=stale_accepted` with lineage status `stale accepted above current best
    9.5078322706`.

Decision:

- Keep the implementation small and state-based rather than adding another planner phrase ban.
- Future planner context should no longer treat reverted noisy acceptances as the live best merely
  because old attempt JSON still contains `gate accepted=true`.

## 2026-05-10 - Checkpoint 5.25: Score pending V-pipeline transform

Success criteria for this checkpoint:

- Verify the stale-history fix changes planner state enough that it uses the corrected current
  lineage best.
- Resolve the pending compiled V-pipeline transform left at the end of the previous loop.

Evidence:

- `avo agent-plan --lineage ./lineage --knowledge knowledge/ampere.md --cwd . --env-file
  ../avo/.env.local --attempts-dir ./attempts --attempt-limit 128` described the seed as
  `9.51` TFLOPS, not the stale `9.54` TFLOPS syncwarp-removal score, and proposed scoring the
  pending V-pipeline transform.
- One scoped `evolve-once` scored that transform:
  - `timeout 1800 .venv/bin/python -m avo evolve-once --lineage ./lineage --knowledge
    knowledge/ampere.md --cwd . --env-file ../avo/.env.local --timeout-s 900
    --compile-repair-attempts 4 --attempts-dir ./attempts --attempt-limit 128 --step-json
    attempts/evolve_once_score_pending_v_pipeline_20260510T1227Z.json`
- The transform hoisted `v_frag` before the PV output-chunk loop and loaded the next V fragment at
  the end of each chunk.
- Result:
  - all 8 full-target BF16 cases correct;
  - candidate geomean `9.304493152841513` TFLOPS;
  - current best geomean `9.507832270603132` TFLOPS;
  - gate rejected with reason `candidate regressed geomean throughput`.

Decision:

- Do not keep the V-pipeline transform. It is a coherent semantic move, but the longer live
  `v_frag` lifetime does not improve the current MMA seed.
- The planner-context fix is doing its job: the next action was based on the corrected lineage best
  and the pending-transform follow-up, not the reverted noisy score.

## 2026-05-10 - Checkpoint 5.26: Force score after successful compile-only transforms

Success criteria for this checkpoint:

- Run a fresh loop with stale-history correction active.
- Identify whether compile-only transforms are reliably followed by score validation.
- Fix the simplest orchestration gap without adding a CUDA-specific ban.

Research refresh:

- Exa search for Ampere `cp.async`/FlashAttention guidance surfaced NVIDIA CUTLASS
  `flash_attention_v2.py`, the NVIDIA Ampere tuning guide, and CUDA async-copy documentation.
  The useful takeaway remains structural: useful Ampere async-copy work is a coherent
  producer/consumer pipeline with aligned tiled copies, tensor-core consumption, and register/shared
  staging, not a scalar copy-size tweak.

Loop:

- Command:
  - `timeout 43200 .venv/bin/python -m avo evolve-loop --lineage ./lineage --knowledge
    knowledge/ampere.md --cwd . --env-file ../avo/.env.local --timeout-s 10800
    --max-steps 16 --compile-repair-attempts 4 --attempts-dir ./attempts --attempt-limit 160
    --loop-json attempts/loop_after_stale_history_fix_20260510T1232Z.json`
- Result:
  - `accepted=false`
  - `completed_steps=16`
  - `stopped_reason=max_steps`

Evidence:

- All scored candidates were correct but slower than the current `9.507832270603132` best:
  - removed score-store `__syncwarp()`: `9.409806486510405`;
  - `kThreads=128`: `9.06660586019657`;
  - another removed-score-store-`__syncwarp()` attempt: `9.360119168246715`;
  - cooperative Q-fragment load/register change: `8.046233836802045`;
  - `kThreads=80`: `9.079055906599761`.
- The loop also produced seven successful compile-only transforms that were not immediately scored:
  cooperative K shared staging, multi-query-tile work mapping, repaired Q async double buffering,
  another K shared-staging variant, another K shared-tile variant, K double buffering, and
  K-register pipelining.
- This exposed a remaining orchestration bug: the previous normalizer only rewrote an exact
  repeated compile into a score. The planner could avoid the pending score obligation by proposing
  a different compile-only transform or a different score transform.
- One Q async double-buffer compile required repair and succeeded, so compile self-repair is still
  working. The issue is not repair; it is unresolved compile-only state accumulating.

Change:

- Runtime pending-transform normalization now rewrites any following compile decision into a score
  of the pending transform when the seed family can be inferred.
- Runtime attempt-history validation now rejects moving to another compile, another score transform,
  or another command while a semantic compile-only transform is pending. The only valid follow-up is
  a score carrying the same `candidate_transform`.

Verification:

- Focused tests:
  - `.venv/bin/python -m pytest tests/test_cli.py::test_pending_transform_payload_normalizer_rewrites_repeated_mma_compile_to_score tests/test_cli.py::test_pending_transform_payload_normalizer_forces_score_before_new_mma_compile tests/test_evolve.py::test_attempt_history_rejects_new_compile_while_transform_score_pending tests/test_evolve.py::test_attempt_history_rejects_different_score_while_transform_score_pending tests/test_evolve.py::test_attempt_history_rejects_score_without_pending_transform -q`: passed, 5 tests.
- Affected suites:
  - `.venv/bin/python -m pytest tests/test_cli.py tests/test_evolve.py -q`: passed, 122 tests.
- Lint and patch hygiene:
  - `.venv/bin/ruff check avo/cli.py avo/evolve.py tests/test_cli.py tests/test_evolve.py`: passed.
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 400 tests.

Decision:

- Keep this as an orchestration invariant, not a prompt note: once a semantic transform compiles,
  the next useful action is to score that same transform or reject it structurally.
- Do not add another family ban from this loop. The useful fix is making the search loop resolve
  each compiled semantic move before exploring another one.

## 2026-05-10 - Checkpoint 5.27: Bound Anthropic planner request latency

Success criteria for this checkpoint:

- Validate the forced pending-score fix with a short live loop, or identify the next reliability
  blocker if the loop does not reach a compile/score transition.
- Ensure a slow planner API call cannot hold the evolve loop indefinitely.
- Clean up any generated candidate residue from interrupted validation.

Validation attempt:

- Command:
  - `timeout 14400 .venv/bin/python -m avo evolve-loop --lineage ./lineage --knowledge
    knowledge/ampere.md --cwd . --env-file ../avo/.env.local --timeout-s 3600
    --max-steps 4 --compile-repair-attempts 4 --attempts-dir ./attempts --attempt-limit 180
    --loop-json attempts/loop_after_forced_pending_score_20260510T1319Z.json`
- Result:
  - No step record was emitted.
  - The process idled in the planner path long enough to stop the run manually.
  - The interrupted run left a malformed generated K-staging patch in
    `candidates/cuda_mma_attention/attention_kernel.cu`; it was reverted manually with
    `apply_patch`, and the runtime worktree returned to only intentional code edits.

Change:

- Runtime `avo/agent.py` now adds a per-request Anthropic timeout to planner requests.
- Default timeout is `180` seconds.
- `AVO_AGENT_REQUEST_TIMEOUT_S` can override the default when shorter or longer planner waits are
  needed.
- Invalid, zero, or negative override values fall back to the default.

Verification:

- Local SDK check:
  - installed `anthropic` is `0.100.0`;
  - `Anthropic(..., timeout=...)` and `messages.create(..., timeout=...)` are supported by the
    installed signatures.
- Planner-only smoke:
  - `AVO_AGENT_REQUEST_TIMEOUT_S=45 timeout 180 .venv/bin/python -m avo agent-plan --lineage
    ./lineage --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --attempts-dir
    ./attempts --attempt-limit 180`
  - returned a structured transform proposal without dirtying the tree.
- Focused tests:
  - `.venv/bin/python -m pytest tests/test_agent.py::test_agent_request_timeout_has_env_override tests/test_agent.py::test_decision_request_retries_transient_api_error -q`: passed, 2 tests.
- Agent suite:
  - `.venv/bin/python -m pytest tests/test_agent.py -q`: passed, 196 tests.
- Lint and patch hygiene:
  - `.venv/bin/ruff check avo/agent.py tests/test_agent.py`: passed.
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 401 tests.

Decision:

- Keep this as a request-boundary reliability fix. Do not add signal/kill cleanup in this
  checkpoint; that is broader because SIGTERM can interrupt any point between patch application and
  cleanup.
- A future validation loop can use `AVO_AGENT_REQUEST_TIMEOUT_S=45` or another bounded value when
  testing orchestration behavior.

## 2026-05-10 - Checkpoint 5.28: Make CUDA preflight guidance less brittle

Success criteria for this checkpoint:

- Validate that the bounded planner timeout lets a short loop reach compile/score transitions.
- Keep hard structural preflights for invalid CUDA/search-loop states.
- Move async-copy granularity from hard rejection territory into soft, structured guidance.
- Preserve planner visibility into the warning so the agent can learn from it without losing the
  compile/repair path.

Validation loop:

- Command:
  - `AVO_AGENT_REQUEST_TIMEOUT_S=45 timeout 7200 .venv/bin/python -m avo evolve-loop --lineage
    ./lineage --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --timeout-s
    3600 --max-steps 4 --compile-repair-attempts 4 --attempts-dir ./attempts --attempt-limit 180
    --loop-json attempts/loop_after_forced_pending_score_timeout_20260510T1331Z.json`
- Result:
  - The loop completed all 4 requested steps and stopped at `max_steps`.
  - It correctly alternated compile then score for the pending semantic transform twice.
  - `mma_k_fragment_prefetch_v1` compiled, then scored correct but regressed to
    `3.6639716243616083` geomean TFLOPS, so it was rejected.
  - `mma_q_fragment_coop_load_v1` compiled, then scored correct but regressed to
    `9.40853995478011` geomean TFLOPS against the current `9.507832270603132` best, so it was
    rejected.
  - No kernel was accepted.

CUDA guidance source refresh:

- NVIDIA CCCL/libcu++ `cuda::memcpy_async` docs say Ampere+ may use `cp.async` when data is
  aligned to at least 4 bytes and the copy is global-to-shared.
- The same docs show 16-byte aligned `cuda::aligned_size_t<16>` examples for stronger async-copy
  shape.
- NVIDIA pipeline docs emphasize producer acquire, async submission, producer commit, consumer
  wait/release, converged commits, and valid producer/consumer lifecycle.

Change:

- Runtime `avo/agent.py` now has `CUDA_STRUCTURAL_ADVISORY_TRACKS` alongside the hard
  `CUDA_STRUCTURAL_PREFLIGHT_TRACKS`.
- Narrow async-copy sizes such as scalar BF16 `__pipeline_memcpy_async(...,
  sizeof(__nv_bfloat16))` now trigger `async_copy_granularity_preference` as an advisory, not a
  rejection.
- Runtime `PatchResult` now records `advisories`, and attempt summaries include them when a patch
  applies. This gives the next planner turn useful signal without blocking compile repair.

Verification:

- Focused tests:
  - `.venv/bin/python -m pytest tests/test_agent.py::test_structural_preflight_allows_scalar_bf16_async_copy_for_repair tests/test_agent.py::test_structural_advisory_flags_scalar_bf16_async_copy_without_rejecting tests/test_agent.py::test_structural_preflight_allows_non_scalar_async_copy_size_expression`: passed, 3 tests.
  - `.venv/bin/python -m pytest tests/test_evolve.py::test_run_decision_command_records_soft_cuda_advisories tests/test_evolve.py::test_run_decision_command_preflights_materialized_cuda_transform`: passed, 2 tests.
- Affected suites:
  - `.venv/bin/python -m pytest tests/test_agent.py`: passed, 197 tests.
  - `.venv/bin/python -m pytest tests/test_evolve.py`: passed, 86 tests.
- Lint and patch hygiene:
  - `.venv/bin/python -m ruff check avo/agent.py avo/evolve.py tests/test_agent.py tests/test_evolve.py`: passed.
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest`: passed, 403 tests.

Decision:

- Keep hard rejection for invalid structural CUDA states: malformed transform materialization,
  unsupported WMMA contracts, stale symbols, no-effect helpers/buffers, invalid shared-tile scope,
  and invalid async pipeline lifecycle.
- Treat async-copy width as a performance advisory unless it is tied to an actual structural
  invalidity. The agent should be able to compile/repair coherent async-copy hypotheses, then learn
  from score and compiler evidence.

## 2026-05-10 - Checkpoint 5.29: Drain compiled transforms at loop step limit

Problem:

- The longer evolve loop after soft CUDA advisories ended cleanly at `max_steps`, but the final
  step was a successful compile-only structured transform.
- That left the candidate unresolved until a future loop, which violates the search-loop invariant:
  a coherent semantic move that compiles should be scored or explicitly rejected before the loop
  considers the attempt sequence resolved.

Change:

- Runtime `avo/cli.py` now records loop steps through one helper and, when the normal loop would
  stop at `max_steps`, checks for a pending compile-only structured transform.
- If one exists and its candidate seed path has a known validation command, the loop creates a
  deterministic score decision for the exact same `candidate_transform` and records that extra
  drain step.
- The drain score still goes through `run_decision_command`, `finalize_attempt`, cleanup, and the
  existing compile/correctness repair loop. It does not ask the planner to rediscover the pending
  transform.

Verification:

- Focused regression:
  - `.venv/bin/python -m pytest tests/test_cli.py::test_evolve_loop_scores_pending_compile_transform_at_step_limit tests/test_cli.py::test_evolve_loop_runs_until_accepted_and_records_attempts tests/test_cli.py::test_pending_transform_payload_normalizer_rewrites_repeated_mma_compile_to_score -q`: passed, 3 tests.
- Affected suite:
  - `.venv/bin/python -m pytest tests/test_cli.py -q`: passed, 38 tests.
- Lint:
  - `.venv/bin/python -m ruff check avo tests`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 404 tests.

Decision:

- Keep normal planner-driven behavior during the requested step budget.
- Treat the deterministic drain as completion of an already-started candidate evaluation, not as a
  new planner budget step. This keeps `max_steps` from splitting compile and score across separate
  operator runs.

## 2026-05-10 - Checkpoint 5.30: Require confirmation before one-shot acceptance

Live validation loop:

- Command:
  - `AVO_AGENT_REQUEST_TIMEOUT_S=90 timeout 21600 .venv/bin/python -m avo evolve-loop --lineage
    ./lineage --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --timeout-s
    3600 --max-steps 10 --compile-repair-attempts 4 --attempts-dir ./attempts --attempt-limit 260
    --loop-json attempts/loop_after_pending_drain_20260510T1427Z.json`
- Result:
  - The loop finished in 1 recorded step with `accepted=true`.
  - It demonstrated self-repair: two stale-anchor structured transform materialization attempts
    failed, then the repair decision found a materializable transform.
  - The accepted one-shot score added one `__syncthreads()` after Q-fragment loads and reported
    `9.593373728960215` geomean TFLOPS versus the prior `9.507832270603132`.

Follow-up confirmation:

- Confirmation score for the accepted source:
  - `attempts/score_confirm_sync_barrier_20260510T1431Z.json`
  - warmup 3, repeats 3, all 8 cases correct, geomean `9.564407043194905`.
- A/B confirmation for the previous source without that barrier:
  - `attempts/score_confirm_without_q_barrier_20260510T1432Z.json`
  - warmup 3, repeats 3, all 8 cases correct, geomean `9.571011809582503`.
- Interpretation:
  - The accepted one-shot improvement was not robust. The previous source was slightly faster by
    `0.006604766387598104` geomean TFLOPS, about `0.069%`.

Change:

- Reverted the noisy accepted source change in the root runtime worktree.
- Reverted the nested lineage commit `9e24039` with lineage commit `2b5c95e`, restoring latest
  lineage geomean to `9.507832270603132`.
- Runtime `avo/lineage.py` now rejects any one-shot candidate score against an existing candidate
  best for the same benchmark signature. One-shot scores can still establish an initial candidate
  lane, but follow-on improvements must be confirmed with at least warmup 2/repeats 3 before
  acceptance.

Verification:

- Focused lineage tests:
  - `.venv/bin/python -m pytest tests/test_lineage.py -q`: passed, 20 tests.
- Lint:
  - `.venv/bin/python -m ruff check avo/lineage.py tests/test_lineage.py`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 404 tests.

Decision:

- Keep one-shot scores as cheap screening signals only once a candidate best exists.
- Make the planner/loop earn accepted lineage updates with confirmed timing. This is a search-loop
  reliability fix, not a CUDA-specific ban.

## 2026-05-10 - Checkpoint 5.31: Post-confirmation-gate loop run

Command:

- `AVO_AGENT_REQUEST_TIMEOUT_S=90 timeout 14400 .venv/bin/python -m avo evolve-loop --lineage
  ./lineage --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --timeout-s
  3600 --max-steps 6 --compile-repair-attempts 4 --attempts-dir ./attempts --attempt-limit 280
  --loop-json attempts/loop_after_confirmed_gate_20260510T1438Z.json`

Result:

- The loop finished at `max_steps` with `accepted=false`.
- No background evolve, score, compile, worker, `nvcc`, or `python -m avo` process remained after
  the run.
- Step 1 failed planning validation before execution: the planner claimed cooperative K tile
  staging/reduced K-load traffic, but the structured transform did not actually reduce K load sites
  or move them out of the repeated loop.
- Steps 2-5 produced executable score attempts on the large Ampere benchmark signature
  (`seq_lens=4096,8192,16384,32768`, `total_tokens=32768`, `num_heads=16`, `head_dim=128`,
  `bf16`, causal and noncausal).
- All four scored candidates were correct and cleaned up after scoring, but each regressed versus
  the current best `9.507832270603132` geomean TFLOPS:
  - Step 2: `9.070815939784865`.
  - Step 3: `9.077303848372576`.
  - Step 4: `9.31778686584628`.
  - Step 5: `9.331295759794157`.
- Step 5 exercised the repair path: the first scored attempt had one repair, then the repaired
  candidate executed correctly but still regressed.
- Step 6 failed planning validation before execution because the decision itself described an
  out-of-bounds/correctness failure mode for the proposed patch.

Decision:

- The confirmation gate is doing the right thing: no one-shot result was accepted into lineage in
  this run.
- Self-repair is active for executable candidates, but planning still wastes budget on proposals
  whose text and structured transform disagree, or whose own risk text says the patch is invalid.
- Next useful search-loop work is to make those failures produce better follow-up context for the
  planner rather than adding more hard CUDA-specific bans.

## 2026-05-10 - Checkpoint 5.32: Surface planning validation feedback

Problem:

- The post-confirmation-gate loop showed two planning-validation failures:
  - A structured transform claimed K-load traffic reduction but did not make a matching dataflow
    change.
  - A patch was described by its own risk text as likely out-of-bounds/incorrect.
- These were classified, but the planner-facing attempt summary mostly exposed the class and not
  the short reason. That makes the loop less self-correcting than it should be.

Change:

- Runtime `avo/evolve.py` now includes a capped `planning_feedback=...` field in recent-attempt
  summaries for planning-validation failures.
- Predicted correctness failures from planning-risk validation now get their own class,
  `planning_predicted_correctness_failure`, instead of being folded into generic
  `planning_validation`.
- This is intentionally not another CUDA hard ban. It gives the next planner turn concise feedback
  about why the proposed edit was invalid so it can repair the semantic move.

Verification:

- Focused tests:
  - `.venv/bin/python -m pytest tests/test_evolve.py::test_summarize_attempt_history_classifies_transform_semantic_mismatch tests/test_evolve.py::test_summarize_attempt_history_classifies_predicted_correctness_planning_failure -q`:
    passed, 2 tests.
- Affected suite:
  - `.venv/bin/python -m pytest tests/test_evolve.py -q`: passed, 87 tests.
- Lint and patch hygiene:
  - `.venv/bin/python -m ruff check avo tests`: passed.
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 405 tests.

Decision:

- Keep planner validation hard for decisions that contradict themselves or describe a known
  correctness break.
- Make the recovery path softer and more useful: feed the concise failure reason into the next
  planning context rather than expanding a reactive CUDA-specific ban list.

## 2026-05-10 - Checkpoint 5.33: Promote only concrete hard preflight tracks

Problem:

- Planning feedback classes are useful context, but they are not always enforceable as hard
  preflight checks.
- Treating every recurring class as hard-promotable overclaims what the loop can actually enforce.
  That was too close to the earlier "advice to the planner" failure mode.

Change:

- Runtime `avo/evolve.py` now persists promoted preflight state only when the recurring failure
  class has concrete structural preflight checks.
- Supervisor summaries still report recurring planning classes, but now say when no concrete hard
  preflight track exists and route the signal back into planner feedback instead of claiming hard
  enforcement.
- Existing concrete CUDA classes such as stale symbols, syntax errors, and unsupported WMMA shapes
  still promote to their structural preflight checks.

Verification:

- Focused tests:
  - `.venv/bin/python -m pytest tests/test_evolve.py::test_planning_feedback_class_is_not_promoted_without_concrete_preflight tests/test_evolve.py::test_update_promoted_preflight_tracks_persists_recurring_class -q`:
    passed, 2 tests.
- Affected suite:
  - `.venv/bin/python -m pytest tests/test_evolve.py -q`: passed, 88 tests.
- Lint and patch hygiene:
  - `.venv/bin/python -m ruff check avo tests`: passed.
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 406 tests.

Decision:

- Keep a clear boundary: concrete recurring CUDA failure classes can become hard structural
  preflights; planning feedback classes remain planner-memory unless/until there is a real check to
  enforce.

## 2026-05-10 - Checkpoint 5.34: Loop after planning feedback and concrete promotion fixes

Command:

- `AVO_AGENT_REQUEST_TIMEOUT_S=90 timeout 21600 .venv/bin/python -m avo evolve-loop --lineage
  ./lineage --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --timeout-s
  3600 --max-steps 8 --compile-repair-attempts 4 --attempts-dir ./attempts --attempt-limit 320
  --loop-json attempts/loop_after_planning_feedback_20260510T1510Z.json`

Source check:

- Exa/NVIDIA source check before logging:
  - NVIDIA Ampere Tuning Guide:
    `https://docs.nvidia.com/cuda/ampere-tuning-guide/`
  - CUDA Programming Guide async copies:
    `https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/04-special-topics/async-copies.html`
- Relevant constraints:
  - For compute capability 8.6, maximum concurrent warps per SM is 48, register file is 64K
    32-bit registers per SM, maximum thread blocks per SM is 16, and shared memory capacity is
    100 KB per SM / 99 KB per block.
  - Ampere adds async global-to-shared copy support, but ordinary shared-memory staging still has
    to earn its cost against coalescing, barriers, bank behavior, and occupancy.
  - LDGSTS/cp.async supports 4/8/16-byte element copies, with best performance requiring stronger
    alignment; this supports keeping async-copy width as advisory unless the transform is
    structurally invalid.

Result:

- The loop finished at `max_steps` with `accepted=false`.
- No background evolve, score, compile, worker, `nvcc`, or `python -m avo` process remained after
  the run.
- The runtime tree, docs tree, and nested lineage tree were clean afterward.
- Step 1 compiled a V shared-memory staging transform successfully for `sm_86`.
- Step 2 scored the same V staging transform because the pending-score invariant forced the
  compile-only transform to be evaluated. It was correct but regressed to
  `3.8717621480649687` geomean TFLOPS versus current best `9.507832270603132`.
- Steps 3-5 were executable one-shot scores, all correct and cleaned up, but all regressed:
  - Step 3: `9.158307794702491`.
  - Step 4: `9.254239332890148`.
  - Step 5: `8.749709748531057`.
- Step 6 failed planning validation after two repair attempts. The recorded failure was a Q-load
  semantic mismatch: the text claimed reduced/reused Q loads, but the structured transform did not
  reduce Q load sites or move them out of the repeated loop.
- Step 7 was rejected by structural preflight as `no_effect_or_skeleton`: a pragma-only unroll edit
  does not change dataflow and must be paired with a substantive transform.
- Step 8 scored `kThreads=128`; it was correct and cleaned up but regressed to
  `9.224937902194656`.

Interpretation:

- The latest loop fixes are working structurally:
  - compiled transforms are not left pending;
  - scored regressions are cleaned up;
  - planning feedback is visible to the loop;
  - no planning-only class was promoted as if it had a concrete hard preflight;
  - no raw CUDA diff path was used for kernel evolution.
- The planner still overstates semantic value in some proposals. It improved enough to produce
  executable candidates, but it still spent budget on shared-memory staging that did not beat the
  current kernel and on one no-effect pragma-only transform.
- The observed V-staging regression is plausible: adding a cooperative shared-memory load and
  barriers can lose to already coalesced/cached global V fragment loads in this simple one-warp
  WMMA seed. The source check supports treating this as an optimization tradeoff, not a structural
  invalidity.

Decision:

- Do not preserve any candidate from this run.
- Keep V shared-memory staging and `kThreads=128` as measured regressions in attempt memory.
- Next useful improvement is planner-side: favor transforms with verifiable semantic deltas in the
  source over generic "stage into shared memory" or pragma-only tuning claims.

## 2026-05-10 - Checkpoint 5.35: Linewise insert materialization

Problem:

- The planning-feedback loop showed a structured transform with a coherent V-staging intent, but
  the materialized diff glued inserted text onto existing CUDA lines:
  - `old_scale[kTile];  __shared__ ...`
  - `for (...) {    for (...) { ... }`
- The candidate happened to compile, but this is a poor interface for kernel evolution. A scoped
  `insert_after_once` or `insert_before_once` should produce a reviewable linewise edit even when
  the planner omits a leading or trailing newline in the inserted text.

Change:

- Runtime `avo/evolve.py` now normalizes `insert_before_once` and `insert_after_once` materialized
  text at the insertion boundary:
  - add a leading newline when inserting after a line-fragment anchor;
  - add a trailing newline when inserting before a line-fragment anchor;
  - preserve explicit newlines when the planner already supplied them.
- This is a transform-interface hardening change, not a CUDA guard. It makes small semantic moves
  reviewable and recoverable without requiring the planner to hand-format raw diff whitespace.

Verification:

- Focused tests:
  - `.venv/bin/python -m pytest tests/test_evolve.py::test_materialize_candidate_transform_inserts_after_anchor_on_new_line tests/test_evolve.py::test_materialize_candidate_transform_inserts_before_anchor_on_own_line -q`:
    passed, 2 tests.
- Affected suite:
  - `.venv/bin/python -m pytest tests/test_evolve.py -q`: passed, 90 tests.
- Lint and patch hygiene:
  - `.venv/bin/python -m ruff check avo tests`: passed.
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 408 tests.

Decision:

- Keep `replace_once` exact because it is intentionally a precise source transformation.
- Make insert transforms linewise by default because most CUDA/Python source insertions are
  statement/block insertions, and line-glued materialization is almost never the intended semantic
  move.

## 2026-05-10 - Checkpoint 5.36: Source-verifiable semantic transform claims

Problem:

- The latest loop still spent budget on proposals whose planning text overstated the transform:
  a Q-load reuse claim without a matching source delta, and generic staging/tuning descriptions
  that were not always tied tightly to changed load/store/loop sites.
- Existing validators catch some mismatch classes, but the prompt should make the rule explicit
  before the planner emits the candidate.

Change:

- Runtime `avo/agent.py` now tells the planner that the claimed semantic delta must be
  source-verifiable from `candidate_transform`.
- The prompt and repo context now say that claims about fewer/reused loads, moved staging,
  different work mapping, or pipeline overlap must correspond to exact transform steps that
  remove, replace, or relocate the relevant current load/store/loop sites.
- Semantic-mismatch feedback now repeats the same rule so repair turns have a concrete target.

Verification:

- Focused prompt tests:
  - `.venv/bin/python -m pytest tests/test_agent.py::test_build_repo_context_lists_local_candidates tests/test_agent.py::test_build_variation_prompt_includes_repo_context tests/test_agent.py::test_decision_feedback_explains_transform_semantic_mismatch_error -q`:
    passed, 3 tests.
- Affected suite:
  - `.venv/bin/python -m pytest tests/test_agent.py -q`: passed, 197 tests.
- Lint and patch hygiene:
  - `.venv/bin/python -m ruff check avo tests`: passed.
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 408 tests.

Decision:

- Keep this as guidance and feedback, not another narrow hard CUDA ban. Existing validators still
  reject objective mismatches; the prompt should reduce the chance that the planner emits them.

## 2026-05-10 - Checkpoint 5.37: Source-verifiable prompt loop and score-environment repair fix

Command:

- `AVO_AGENT_REQUEST_TIMEOUT_S=90 timeout 14400 .venv/bin/python -m avo evolve-loop --lineage
  ./lineage --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --timeout-s
  3600 --max-steps 6 --compile-repair-attempts 4 --attempts-dir ./attempts --attempt-limit 340
  --loop-json attempts/loop_after_source_verifiable_prompt_20260510T1550Z.json`

Loop result:

- The loop finished at `max_steps` with `accepted=false`.
- No background evolve, score, compile, worker, `nvcc`, or `python -m avo` process remained after
  the run.
- Step 1 compiled `mma_q_hoist_sync_v1`. This was a source-verifiable transform shape, but the
  actual materialized patch only added a `__syncthreads()` after already-hoisted Q fragment loads,
  so it did not implement the stated Q-hoist semantic move.
- Step 2 was rejected by planning validation because the planner described its own transform as a
  predicted correctness failure.
- Step 3 compiled `mma_q_shared_coop_v2`, which added shared Q tile storage, a cooperative Q load
  before the key loop, and changed the WMMA Q fragment source to shared memory.
- Step 4 scored that Q shared-memory staging transform with confirmed timing (`warmup=2`,
  `repeats=3`). It was correct but regressed to `9.19933585352861` geomean TFLOPS versus current
  best `9.507832270603132`.
- Step 5 compiled `mma_remove_score_sync_v1`, a concrete synchronization-removal transform.
- Step 6 failed planning validation after one repair attempt. The repair attempt scored a
  `kThreads=64` transform, but the score payload reported `RuntimeError: CUDA is not available`
  for every case even though `nvidia-smi` and a follow-up `avo env` showed CUDA healthy
  immediately after the loop.

Change:

- Runtime `avo/evolve.py` now treats score-environment failures as non-repairable by the
  correctness self-repair path.
- `score_environment_error` attempts still appear in attempt summaries, but the loop no longer
  asks the planner to repair the CUDA candidate as if the kernel produced wrong math.

Verification:

- Post-loop health checks:
  - `nvidia-smi --query-gpu=name,compute_cap,utilization.gpu,memory.used --format=csv,noheader`:
    `NVIDIA RTX A6000, 8.6, 0 %, 3 MiB`.
  - `.venv/bin/python -m avo env`: `torch.cuda_available=true`, target `sm_86`.
- Focused tests:
  - `.venv/bin/python -m pytest tests/test_evolve.py::test_score_environment_error_is_not_repairable_correctness_failure tests/test_evolve.py::test_summarize_attempt_history_classifies_score_environment_error -q`:
    passed, 2 tests.
- Affected suite:
  - `.venv/bin/python -m pytest tests/test_evolve.py -q`: passed, 91 tests.
- Lint and patch hygiene:
  - `.venv/bin/python -m ruff check avo tests`: passed.
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 409 tests.

Decision:

- Keep the source-verifiable prompt rule. It improved the shape of emitted transforms, though it
  does not guarantee the planner's prose perfectly matches the materialized edit.
- Treat transient CUDA availability and build-environment score failures as orchestration
  substrate failures, not correctness failures requiring CUDA source repair.

## 2026-05-10 - Checkpoint 5.38: Source-verifiable load-claim validation fix

Finding:

- The semantic validator intended to catch claims like "reduce/reuse/hoist Q loads" had a Python
  f-string bug: regex quantifiers were written as `{0,80}` inside an f-string, so the pattern
  searched for the literal tuple-like text `(0, 80)` instead of a bounded wildcard.
- Replaying step 1 from
  `attempts/loop_after_source_verifiable_prompt_20260510T1550Z.json` confirmed the practical
  impact: the planner claimed a Q-hoist out of `key_start`, but the materialized transform only
  inserted `__syncthreads()` after Q loads that were already outside the `key_start` loop.

Change:

- Runtime `avo/agent.py` now escapes the regex quantifier and recognizes load-reduction wording
  such as remove/move/relocate in addition to reduce/reuse/hoist.
- Load-relocation validation now extracts named loop variables from claims such as "outside the
  key_start loop" and requires the transform to actually move the operand load from before a
  matching loop-header context to outside that named loop.
- This is still a semantic source-verification check, not a phrase ban: a real Q-load hoist out
  of `key_start` is accepted, while a sync-only transform with the same claim is rejected.

Verification:

- Focused tests:
  - `.venv/bin/python -m pytest tests/test_agent.py -q -k "q_load_hoist or q_hoist_claim or claimed_v_reuse or probability_load_hoist or current_q_reuse"`:
    passed, 5 tests.
- Replay check:
  - Parsing the historical `mma_q_hoist_sync_v1` decision now raises
    `candidate_transform semantic mismatch` for the Q-load reduction claim.
- Affected suites and hygiene:
  - `.venv/bin/python -m pytest tests/test_agent.py -q`: passed, 199 tests.
  - `.venv/bin/python -m pytest tests/test_evolve.py -q`: passed, 91 tests.
  - `.venv/bin/ruff check avo tests`: passed.
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 411 tests.

Decision:

- Prefer this structural semantic check over adding another historical "do not repeat Q-hoist
  sync" instruction. The planner can still make Q-load hoist attempts, but the transform must now
  prove the claimed loop/dataflow movement in the source snippets.

## 2026-05-10 - Checkpoint 5.39: Partial post-load-claim loop and exact-repeat memory

Command:

- `AVO_AGENT_REQUEST_TIMEOUT_S=120 timeout 21600 .venv/bin/python -m avo evolve-loop --lineage
  ./lineage --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --timeout-s
  3600 --max-steps 10 --compile-repair-attempts 4 --attempts-dir ./attempts --attempt-limit 380
  --loop-json attempts/loop_after_load_claim_validation_20260510T1605Z.json`

Loop result:

- The loop process was interrupted before it wrote the requested final loop JSON, so this is a
  partial validation run, not a completed loop.
- It produced six per-attempt records from `2026-05-10T16-08-44-00-00.json` through
  `2026-05-10T16-22-34-00-00.json`.
- Post-interruption checks showed no remaining evolve, score, compile, worker, `nvcc`, or `ninja`
  process. The runtime repo and lineage repo were clean. The A6000 was healthy and idle:
  `NVIDIA RTX A6000, 8.6, 0 %, 3 MiB`.

Partial attempts:

- `kThreads=80`: correct, regressed to `9.23723076059812` geomean TFLOPS.
- `kThreads=64`: correct, regressed to `9.160553963665484`.
- Move `output_acc` rescaling into the active-warp guard: correct, regressed badly to
  `1.5832549312452529`.
- `kThreads=64` again: correct, regressed to `9.018591018326017`.
- `kThreads=128`: correct, regressed to `9.15970458759749`.
- `kThreads=80` again: correct, regressed to `9.103867860816868`.

Finding:

- The source-verifiable load-claim fix prevented the previously observed fake Q-hoist pattern in
  this partial run.
- The remaining weakness was search-loop memory: the planner repeated exact already-scored
  regressed transforms, especially `kThreads=80` and `kThreads=64`, instead of treating those
  scored regressions as exhausted candidates.

Change:

- Runtime `avo/evolve.py` now rejects a decision whose `candidate_transform` exactly matches a
  previously scored, unaccepted transform.
- The check intentionally allows retry when the previous score failed as `score_environment_error`;
  transient CUDA/build substrate failures should not permanently poison a candidate.

Verification:

- Focused tests:
  - `.venv/bin/python -m pytest tests/test_evolve.py -q -k "repeated_compile_only_transform or repeated_scored_unaccepted_transform or repeated_transform_after_score_environment_error"`:
    passed, 3 tests.
- Affected suite and hygiene:
  - `.venv/bin/python -m pytest tests/test_evolve.py -q`: passed, 93 tests.
  - `.venv/bin/ruff check avo tests`: passed.
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 413 tests.

Decision:

- This is not a CUDA phrase ban or a hard family ban. It is attempt-memory hygiene: exact scored
  regressions should not be proposed again unless the prior score was an environment failure.

## 2026-05-10 - Checkpoint 5.40: Pending-transform history ordering and repair context

Commands:

- `AVO_AGENT_REQUEST_TIMEOUT_S=120 timeout 14400 .venv/bin/python -m avo evolve-loop --lineage
  ./lineage --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --timeout-s
  3600 --max-steps 6 --compile-repair-attempts 4 --attempts-dir ./attempts --attempt-limit 420
  --loop-json attempts/loop_after_repeat_guard_20260510T1636Z.json`
- `AVO_AGENT_REQUEST_TIMEOUT_S=120 timeout 7200 .venv/bin/python -m avo evolve-loop --lineage
  ./lineage --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --timeout-s
  3600 --max-steps 1 --compile-repair-attempts 4 --attempts-dir ./attempts --attempt-limit 420
  --loop-json attempts/loop_after_attempt_order_fix_20260510T1859Z.json`

Loop result:

- The repeat-transform guard fired in the 6-step loop: one planner attempt was rejected with
  `candidate_transform repeats a previously scored unaccepted transform`.
- The loop also showed useful semantic validation: two later planner attempts were rejected for
  Q-load reuse claims whose structured transform did not actually reduce or move Q load sites.
- Compile self-repair worked: a multi-query-tile-per-block transform first failed with
  `identifier "query_start" is undefined`; the repair moved output normalization/store into the
  `local_tile` loop and compiled successfully.
- The repaired transform then exposed two attempt-memory bugs:
  - pending-transform lookup used filename order, so an older `evolve_once_...json` score sorted
    after timestamped attempt records and masked the newer compile-only pending transform;
  - after scoring a pending transform, correctness-repair validation still read only on-disk
    attempts, so it could reject the repair as if the old compile-only transform were still
    unresolved.

Change:

- Runtime `avo/evolve.py` now orders attempt records by `attempt.completed_at` / `started_at`
  inside the JSON payload instead of by filename, with file mtime only as fallback.
- `validate_decision_against_attempt_history` now accepts in-memory extra payloads.
- Runtime `avo/cli.py` passes the current failed attempt into repair validation, so a just-scored
  pending transform is considered resolved during correctness repair even before the enclosing
  step has been written.

Live validation:

- After the timestamp-ordering fix, `pending_compile_only_transform(Path("attempts"))` returned
  the real pending batch transform instead of `None`.
- The 1-step validation loop entered `avo score` for that pending transform.
- The candidate failed correctness on all target cases with
  `RuntimeError: candidate output contains non-finite values`, so it was not accepted.
- The resulting attempt record now clears the pending transform from history; a follow-up
  `pending_compile_only_transform(Path("attempts"))` returned `None`.

Verification:

- Focused tests:
  - `.venv/bin/python -m pytest tests/test_cli.py tests/test_evolve.py -q -k "repair_loop_allows_correctness_repair_after_pending_transform_score or pending_compile_transform_uses_payload_timestamps or scores_pending_compile_transform_at_step_limit"`:
    passed, 3 tests.
- Affected suites and hygiene:
  - `.venv/bin/python -m pytest tests/test_evolve.py tests/test_cli.py -q`: passed, 133 tests.
  - `.venv/bin/ruff check avo tests`: passed.
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 415 tests.

Decision:

- Keep the multi-query-tile transform rejected by evidence, not by a new guard. It compiled after
  self-repair, then failed correctness on the realistic validation workload.
- Treat attempt history as chronological data from payload timestamps, not filename convention;
  loop/manual artifact names should not change search memory semantics.

## 2026-05-10 - Checkpoint 5.41: Batch transforms tolerate redundant no-op steps

Sources checked:

- NVIDIA Ampere Tuning Guide:
  `https://docs.nvidia.com/cuda/ampere-tuning-guide/`
  - sm86 has 48 resident warps/SM, 64K 32-bit registers/SM, up to 16 resident
    blocks/SM, 100 KB shared memory/SM, and 99 KB shared memory per block with
    opt-in dynamic shared memory.
  - Ampere async global-to-shared copy can overlap memory movement with compute,
    avoid extra registers, and may bypass L1.
- CUDA C++ Programming Guide, async copies:
  `https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#async-copies`
  - Hardware async copy uses 4, 8, or 16 byte transfers from global to shared
    memory, aligned to the transfer size; 128 byte alignment is the performance
    target when practical.
- FlashAttention v2.8.3 interface:
  `https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/flash_attn/flash_attn_interface.py`
  - FA2 treats sm8x separately from sm80. For head_dim <= 128, sm8x uses
    block_n=64 for causal/no-dropout cases and block_n=32 otherwise.

Loop command:

- `AVO_AGENT_REQUEST_TIMEOUT_S=120 timeout 10800 .venv/bin/python -m avo evolve-loop
  --lineage ./lineage --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local
  --timeout-s 3600 --max-steps 3 --compile-repair-attempts 4 --attempts-dir ./attempts
  --attempt-limit 450 --loop-json attempts/loop_after_history_repair_context_20260510T1903Z.json`

Loop result:

- The loop completed 3 steps without accepting a candidate.
- Step 1 failed planning validation because `candidate_transform` was not an object.
- Step 2 was rejected because the next command repeated an already recorded unpatched MMA seed
  score without a structural candidate change.
- Step 3 proposed a coherent query-tile batch transform, but materialization rejected the batch
  before compile because one step was an identity replacement of shared-memory declarations.

Change:

- Runtime `avo/evolve.py` now skips no-op steps inside `op=batch` transforms.
- Standalone no-op transforms still fail immediately.
- All-no-op batches still fail via the existing final `transform produced no source change` check.

Why:

- This matches the current transform contract better: a batch is the smallest coherent semantic
  move. A redundant identity step inside that move should not invalidate the useful edits around
  it, but a transform that changes nothing is still not executable search progress.

Verification:

- Focused tests:
  - `.venv/bin/python -m pytest tests/test_evolve.py -q -k "noop_batch or duplicate_include or materialize_candidate_transform_generates_batch_patch"`:
    passed, 4 tests.
- Affected suites and hygiene:
  - `.venv/bin/python -m pytest tests/test_evolve.py tests/test_cli.py -q`:
    passed, 135 tests.
  - `.venv/bin/ruff check avo tests`: passed.
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 417 tests.

Decision:

- Do not add a new CUDA guard for this failure. The candidate had a recoverable transform
  materialization issue, not an invalid CUDA hypothesis. Let the batch reach compile/correctness
  when it contains at least one real source change.

## 2026-05-10 - Checkpoint 5.42: Pending scores survive older repair scores

Sources checked:

- RepairAgent, `https://arxiv.org/pdf/2403.17134`
  - Useful pattern: keep a dynamic prompt with the last executed command/result and let the
    agent interleave information gathering, repair attempts, and validation.
  - Tool-mediated repair should apply a candidate, run validation, and revert failed attempts so
    the next attempt starts from clean source.
- LLMLOOP, `https://www.arxiv.org/pdf/2603.23613`
  - Useful pattern: every generated code change re-enters the compilation loop before later
    validation stages. Compilation and test failures are fed back as structured context.
- CYCLE, `https://arxiv.org/abs/2403.18746`
  - Useful reminder: self-refinement depends heavily on the quality of execution feedback; the
    system should preserve the exact failed edit and validation result, not drown repair in stale
    unrelated failures.

Loop command:

- `AVO_AGENT_REQUEST_TIMEOUT_S=180 timeout 14400 .venv/bin/python -m avo evolve-loop
  --lineage ./lineage --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local
  --timeout-s 3600 --max-steps 6 --compile-repair-attempts 4 --attempts-dir ./attempts
  --attempt-limit 500 --loop-json attempts/loop_after_noop_batch_tolerance_20260510T1913Z.json`

Loop result:

- The first three steps repeated a previously scored unaccepted transform and were rejected by
  attempt-memory validation.
- Step 4 escaped that repetition. A repair path produced a new V-fragment prefetch transform that
  compiled successfully as `build/mma_v_pipeline_prefetch_v1`.
- The final compile-only transform stayed unresolved because pending-transform detection treated
  older score payloads inside the same step's `repair_attempts` as if they had scored the newer
  final compile-only transform.
- Steps 5 and 6 then planned unrelated invalid transforms instead of draining the pending compile
  into a score.

Change:

- Runtime `avo/evolve.py` now identifies scored transforms by matching the score attempt's
  `candidate_transform` identity, including repair attempts.
- Pending compile detection now checks whether the final attempt in a step is a successful
  compile-only transform before letting older repair score payloads clear pending state.

Validation run:

- `AVO_AGENT_REQUEST_TIMEOUT_S=120 timeout 7200 .venv/bin/python -m avo evolve-loop
  --lineage ./lineage --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local
  --timeout-s 3600 --max-steps 1 --compile-repair-attempts 2 --attempts-dir ./attempts
  --attempt-limit 500 --loop-json attempts/loop_after_pending_score_order_fix_20260510T1931Z.json`

Validation result:

- The loop correctly scored the pending V-fragment prefetch transform.
- The candidate was correct on all 8 validation cases.
- It regressed geomean throughput from `9.507832270603132` to `9.419589865765866` TFLOPS, so the
  gate rejected it and cleanup reverted the patch.
- A follow-up `pending_compile_only_transform(Path("attempts"))` returned `None`.

Verification:

- Focused tests:
  - `.venv/bin/python -m pytest tests/test_evolve.py -q -k "pending_compile_transform or repeated_compile_only_transform or repeated_scored_unaccepted_transform"`:
    passed, 4 tests.
- Affected suites and hygiene:
  - `.venv/bin/python -m pytest tests/test_evolve.py tests/test_cli.py -q`:
    passed, 136 tests.
  - `.venv/bin/ruff check avo tests`: passed.
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 418 tests.

Decision:

- This was a history-ordering bug, not a CUDA preflight problem. A score should clear a pending
  transform only when the score belongs to that same transform, or when it is a later step in the
  attempt stream.
- Keep the V-fragment prefetch rejected by measurement, not by a new guard: it was correct but
  slower on the target A6000 validation workload.

## 2026-05-10 - Checkpoint 5.43: Planner provider failures are recorded

Sources checked:

- ARCS, `https://arxiv.org/html/2504.20434v2`
  - Useful pattern: run a budgeted synthesize-execute-repair loop, retrieve relevant evidence,
    execute candidates in a sandbox, encode execution feedback for targeted revision, and stop
    under an explicit budget.
- RepairAgent, `https://arxiv.org/pdf/2403.17134`
  - Useful pattern already relevant to AVO: apply a candidate, run validation, revert failed
    attempts, and feed the actual command/test output into the next repair decision.
- Local async-copy knowledge and tests:
  - `knowledge/ampere.md` and `knowledge/b/cuda_programming_practice.md` already treat narrow
    async-copy granularity as an advisory, not a hard rejection. Hard rejections remain for
    structural failures such as invalid async pipeline lifecycle or no-effect helper stubs.

Loop command:

- `AVO_AGENT_REQUEST_TIMEOUT_S=180 timeout 14400 .venv/bin/python -m avo evolve-loop
  --lineage ./lineage --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local
  --timeout-s 3600 --max-steps 8 --compile-repair-attempts 4 --attempts-dir ./attempts
  --attempt-limit 550 --loop-json attempts/loop_after_pending_score_clear_20260510T1936Z.json`

Loop result:

- Step 1 scored a `kThreads=112` retune. It was correct but regressed geomean throughput to
  `8.898204721209686` TFLOPS versus the current best `9.507832270603132`, so the gate rejected
  it and cleanup succeeded.
- Step 2 failed planning validation because the prose claimed reduced or reused Q loads, while
  the executable `candidate_transform` did not make that semantic move.
- Step 3 failed planning validation for the same kind of mismatch around V-load reuse.
- A later step was interrupted before a loop JSON was written because the Anthropic request raised
  a provider error: low credit balance. No evolve, score, compile, `nvcc`, or `ninja` process was
  still running afterward.

Change:

- Runtime `avo/cli.py` now catches exceptions from the planner provider call in the initial
  evolve step and converts them to `planning_failure_step`.
- Runtime `avo/cli.py` does the same for compile/correctness repair planner calls, preserving the
  failed attempt in `repair_attempts` so the orchestrator records what it was trying to repair.
- Validation errors from the planner payload still flow through the existing `ValueError` path.
  Command execution, compile, score, and cleanup failures keep their existing behavior.

Why:

- A provider/API outage is not a CUDA hypothesis failure, but it should still become durable loop
  evidence. The loop should not disappear without a step record after a long run.
- This is consistent with the repair-loop direction: keep the source tree clean, record the exact
  failed action and feedback, and let future budgeted attempts continue from known state.

Verification:

- Focused provider tests:
  - `.venv/bin/python -m pytest tests/test_cli.py -q -k "provider_exception or planner_provider"`:
    passed, 2 tests.
- Affected repair/evolve tests:
  - `.venv/bin/python -m pytest tests/test_cli.py -q -k "evolve_once or repair_loop"`:
    passed, 9 tests.
  - `.venv/bin/python -m pytest tests/test_evolve.py tests/test_cli.py -q`:
    passed, 138 tests.
- Hygiene:
  - `.venv/bin/ruff check avo tests`: passed.
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 420 tests.
- Actual provider-failure smoke:
  - `AVO_AGENT_REQUEST_TIMEOUT_S=60 timeout 900 .venv/bin/python -m avo evolve-loop
    --lineage ./lineage --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local
    --timeout-s 300 --max-steps 1 --compile-repair-attempts 1 --attempts-dir ./attempts
    --attempt-limit 550 --loop-json attempts/loop_after_provider_failure_handling_20260510Tsmoke.json`:
    exited `2` with a normal loop payload.
  - The smoke recorded one planning-failure step for the Anthropic low-credit `BadRequestError`
    and wrote both the step record and loop JSON instead of crashing.

Decision:

- Do not add another CUDA ban or async-copy hard guard for this incident. The failed run exposed
  orchestration fragility, not an invalid kernel transformation.
- Keep async-copy granularity soft: prefer 16-byte vector groups for throughput, but only reject
  async-copy patches when the executable dataflow violates a structural invariant.

## 2026-05-10 - Checkpoint 5.44: Provider outages do not become planner failure memory

Sources checked:

- AgentRx, `https://arxiv.org/pdf/2602.02475`
  - Useful pattern: failed agent trajectories need a failure taxonomy with evidence for the
    critical failure step. For AVO, provider/API failures are an orchestration/tool availability
    class, not a CUDA transformation or planner-contract class.
- Debug2Fix, `https://arxiv.org/abs/2602.18571v1`
  - Useful pattern: better tool design and richer runtime feedback can improve coding-agent
    repair behavior. The feedback should distinguish runtime/program evidence from tool or
    infrastructure failure.

Change:

- Runtime `avo/evolve.py` now classifies planner-provider/API outages as
  `planner_provider_error`.
- `planner_provider_error` is ignored by recurring-failure-class promotion, so repeated API or
  billing outages do not become hard CUDA preflight tracks.
- Repeated identical provider failures also skip repeated-fingerprint supervisor advice. Other
  ignored classes such as generic `command_failed` still keep repeated-fingerprint supervision,
  preserving the existing loop behavior for real command repetition.

Why:

- The previous checkpoint made provider failures durable, but they would have appeared as generic
  `planning_validation`. That risks teaching the loop the wrong lesson: a low-credit Anthropic
  response is not a malformed transform, bad CUDA hypothesis, or planner schema failure.
- Separating this class keeps attempt memory useful without letting infrastructure outages shape
  CUDA search policy.

Verification:

- Focused memory/promotion tests:
  - `.venv/bin/python -m pytest tests/test_evolve.py -q -k "provider_failure or
    planning_feedback or recurring_failure_class or promoted_preflight"`:
    passed, 7 tests.
- Affected suites:
  - `.venv/bin/python -m pytest tests/test_evolve.py tests/test_cli.py -q`:
    passed, 139 tests.
- Hygiene:
  - `.venv/bin/ruff check avo tests`: passed.
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 421 tests.

Decision:

- Keep provider outage records visible in attempt history for auditability, but do not treat them
  as planner/search feedback.
- Do not add CUDA guards for this class. The correct response is orchestration classification and
  retry/budget behavior once Anthropic access is available again.

## 2026-05-10 - Checkpoint 5.45: Accepted source manifests include local dependencies

Sources checked:

- OmniBOR specification, `https://github.com/omnibor/spec/blob/main/spec/SPEC.md`
  - Useful pattern: input manifests list content identifiers for inputs used to build an artifact,
    making dependency changes detectable.
- GitHub npm provenance post, `https://github.blog/security/supply-chain-security/introducing-npm-package-provenance/`
  - Useful pattern: provenance links an artifact back to source inputs and build context. For AVO,
    lineage commits should carry enough source evidence to audit what a candidate score actually
    came from.

Change:

- Runtime `avo/lineage.py` now writes `sources/latest/manifest.json` for accepted source snapshots.
  The manifest records each source path, byte length, and SHA-256 hash.
- Runtime `avo/evolve.py` now expands accepted candidate source snapshots through direct local
  Python imports under `candidates/`, using Python `ast` parsing instead of executing imports.
- Imported local package directories include source files under that package, so helper Python
  modules and companion CUDA/C++ files can be captured when a scored wrapper imports them.
- Existing unchanged-source gate ignores the generated manifest path when comparing latest source
  snapshots, so adding the manifest does not by itself make an unchanged candidate look new.
- README missing-work text now distinguishes this direct local import capture from still-missing
  dynamic import/runtime-discovered dependency capture.

Why:

- AVO lineage commits are the durable memory of accepted candidates. If a scored Python wrapper
  imports a helper outside the scored module and outside the usual companion source directory, the
  previous snapshot could omit a real source input.
- This stays conservative: only direct local imports inside `candidates/` are followed, no arbitrary
  import execution occurs, and dynamic/runtime imports remain out of scope.

Verification:

- Focused source-artifact tests:
  - `.venv/bin/python -m pytest tests/test_evolve.py::test_finalize_attempt_snapshots_scored_candidate_sources_without_patch tests/test_evolve.py::test_finalize_attempt_snapshots_local_python_import_dependencies tests/test_lineage.py::test_commit_score_records_accepted_source_artifacts tests/test_lineage.py::test_commit_score_rejects_unchanged_source_rerun -q`:
    passed, 4 tests.
- Affected suites:
  - `.venv/bin/python -m pytest tests/test_evolve.py tests/test_lineage.py -q`:
    passed, 119 tests.
- Hygiene:
  - `.venv/bin/ruff check avo tests`: passed.
  - `git diff --check`: passed.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 422 tests.

Decision:

- Do not attempt a full Python dependency graph or dynamic import tracer yet. Static direct-local
  import capture plus a content manifest is enough to close the current accepted-artifact audit gap
  without making scoring slower or less deterministic.

## 2026-05-10 - Checkpoint 5.46: Stop evolve loops on planner provider outages

Sources checked:

- Wink: Recovering from Misbehaviors in Coding Agents, `https://www.arxiv.org/pdf/2602.17037`
  - Useful pattern: coding-agent failures should be classified from the trajectory, and targeted
    course-correction should be injected only for failures the agent can recover from. Repeating
    the same failed tool/provider interaction is treated as a misbehavior pattern, not useful
    exploration.
- MemoCoder, `https://arxiv.org/pdf/2507.18812`
  - Useful pattern: a supervisory/mentor layer should distinguish recurring error patterns and
    update repair strategy over time. Persistent failure classes are memory signals, but not every
    class should become a code-preflight guard.

Change:

- Runtime `avo/evolve.py` now exposes `failure_class_for_step(step)` so loop control can make
  decisions from the same classified attempt taxonomy used by attempt summaries.
- Runtime `avo/cli.py` now stops `evolve-loop` with
  `stopped_reason=planner_provider_error` after a recorded planner provider/API outage.
- The loop still writes the failed planning step to `--attempts-dir` and the loop JSON before
  stopping, so the outage remains auditable without spending the rest of a long run budget on the
  same Anthropic/API billing or availability failure.
- README now documents that provider outages stop the loop because repeating them cannot improve
  CUDA search.

Why:

- The previous provider-failure work correctly classified Anthropic/API outages as
  `planner_provider_error` and kept them out of CUDA preflight promotion, but a large
  `--max-steps` loop would still keep calling the provider after the first recorded outage.
- That is the wrong response for an infrastructure class. Compile/correctness failures should go
  through self-repair; provider availability failures should be recorded and stop the current run
  so they do not pollute attempt history or consume budget.

Verification:

- Focused loop-control tests:
  - `.venv/bin/python -m pytest tests/test_cli.py::test_evolve_loop_stops_after_planner_provider_error tests/test_cli.py::test_evolve_loop_records_planning_validation_failure tests/test_cli.py::test_evolve_loop_runs_until_accepted_and_records_attempts -q`:
    passed, 3 tests.
- Affected CLI suite:
  - `.venv/bin/python -m pytest tests/test_cli.py -q`: passed, 42 tests.
- Hygiene:
  - `.venv/bin/ruff check avo tests`: passed.
  - `git diff --check`: passed in both runtime and lab-notebook repos.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 423 tests.

Decision:

- Keep CUDA/search failures repairable inside the evolve loop, but stop immediately on classified
  planner provider/API outages after recording the step. This is a stop policy, not a CUDA guard.

## 2026-05-10 - Checkpoint 5.47: Accepted source snapshots include static extension sources

Sources checked:

- PyTorch `torch.utils.cpp_extension` documentation,
  `https://docs.pytorch.org/docs/2.11/cpp_extension.html`
  - Useful fact: `torch.utils.cpp_extension.load(name, sources=...)` accepts a string or list of
    relative/absolute paths to C++/CUDA source files. Those paths are build inputs, so accepted AVO
    lineage snapshots should capture statically declared local sources when they are under
    `candidates/`.

Change:

- Runtime `avo/evolve.py` now statically parses accepted candidate Python files for
  `sources=` arguments on extension-loading calls.
- The parser resolves conservative `pathlib` expressions such as
  `Path(__file__).resolve().parent / "shared_extension"` and source lists containing
  `str(EXTENSION_DIR / "attention_kernel.cu")`.
- Only existing source files under `candidates/` are captured. The implementation does not import
  candidate modules, execute arbitrary code, follow symlinks, or include files outside the candidate
  source boundary.
- README now states that accepted source snapshots include direct local imports and statically
  declared extension sources, while dynamic/import-executed discovery remains missing.

Why:

- The previous manifest checkpoint captured scored modules, companion directories, and direct local
  imports. That still missed a realistic PyTorch extension pattern where a wrapper lists build
  sources from a shared local directory outside the companion seed directory.
- Capturing statically declared `sources=[...]` closes that audit gap without turning scoring into
  an import tracer.

Verification:

- Focused provenance tests:
  - `.venv/bin/python -m pytest tests/test_evolve.py::test_finalize_attempt_snapshots_static_extension_sources_outside_companion tests/test_evolve.py::test_finalize_attempt_snapshots_local_python_import_dependencies tests/test_evolve.py::test_finalize_attempt_snapshots_scored_candidate_sources_without_patch -q`:
    passed, 3 tests.
- Focused lint:
  - `.venv/bin/ruff check avo/evolve.py tests/test_evolve.py`: passed.
- Affected suites:
  - `.venv/bin/python -m pytest tests/test_evolve.py tests/test_lineage.py -q`:
    passed, 120 tests.
- Hygiene:
  - `.venv/bin/ruff check avo tests`: passed.
  - `git diff --check`: passed in both runtime and lab-notebook repos.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 424 tests.

Decision:

- Keep dependency discovery static and candidate-scoped. Do not execute candidate imports just to
  discover build sources; dynamic source discovery remains a future audit feature.

## 2026-05-10 - Checkpoint 5.48: Budget long-run planner prompts

Sources checked:

- RepairAgent, `https://arxiv.org/pdf/2403.17134`
  - Useful fact: the repair loop uses a dynamically updated prompt with static and dynamic
    sections, including gathered information and the last command/result, so context management is
    part of the agent interface rather than an afterthought.
- Agent trajectory dynamics study, `https://www.arxiv.org/pdf/2506.18824`
  - Useful fact: successful software-agent trajectories respond to prior results and avoid
    repetitive action loops; preserving the newest result/follow-up evidence is more important
    than preserving arbitrary old context verbatim.

Change:

- Runtime `avo/agent.py` now applies a final `MAX_VARIATION_PROMPT_CHARS` budget when building the
  Anthropic variation prompt.
- If dynamic sections would exceed the budget, the prompt compacts bulky repo context, knowledge,
  lineage, and then attempt history.
- Attempt history is tail-preserved so immediate repair requests, supervisor signals, and exact
  pending `candidate_transform` JSON survive long runs.
- Runtime README now documents this prompt-budget behavior.

Why:

- The loop already bounds retrieved knowledge and recent attempt history, but the final Anthropic
  prompt had no hard last-mile budget. Long autonomous runs should degrade by compacting dynamic
  evidence, not by silently constructing an oversized prompt or losing the newest repair signal.
- This follows the same design direction as the repair/self-repair work: keep the latest tool
  result and executable follow-up payload visible, while trimming older context first.

Verification:

- Focused prompt tests:
  - `.venv/bin/python -m pytest tests/test_agent.py::test_build_variation_prompt_compacts_dynamic_sections_with_tail_history tests/test_agent.py::test_build_variation_prompt_includes_repo_context tests/test_agent.py::test_build_variation_prompt_includes_attempt_history -q`:
    passed, 3 tests.
- Affected agent suite:
  - `.venv/bin/python -m pytest tests/test_agent.py -q`: passed, 200 tests.
- Hygiene:
  - `.venv/bin/ruff check avo tests`: passed.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 425 tests.

Decision:

- Keep prompt compaction section-based and deterministic. Do not add another agent, summarizer, or
  learned memory layer until a concrete failure shows the simpler section budget is insufficient.

## 2026-05-10 - Checkpoint 5.49: Prompt-budget loop smoke hit provider stop

Command:

```bash
AVO_AGENT_REQUEST_TIMEOUT_S=120 timeout 1200 .venv/bin/python -m avo evolve-loop \
  --lineage ./lineage \
  --knowledge knowledge/ampere.md \
  --cwd . \
  --env-file ../avo/.env.local \
  --timeout-s 300 \
  --max-steps 5 \
  --compile-repair-attempts 2 \
  --attempts-dir ./attempts \
  --attempt-limit 550 \
  --loop-json attempts/loop_prompt_budget_20260510T214804Z.json
```

Result:

- The loop recorded one planning-failure step and stopped with
  `stopped_reason=planner_provider_error`.
- The Anthropic response was a `BadRequestError` reporting that the credit balance is too low for
  API access.
- No CUDA candidate transform, compile, score, or repair attempt executed because the planner call
  did not produce a decision.

Decision:

- This validates the stop policy under the current account state: provider/account failures are
  infrastructure stops, not CUDA search failures and not reasons to add new preflight guards.
- Resume live evolution when Anthropic credits are available; the loop command shape is otherwise
  ready for a longer run.

## 2026-05-10 - Checkpoint 5.50: Budget invalid-decision retry feedback

Sources checked:

- Local reference agent: `pi-mono-agent/packages/agent/README.md` and
  `pi-mono-agent/packages/agent/src/agent.ts`
  - Useful pattern: context transforms happen before provider conversion, and lifecycle/tool hooks
    preserve a clean transcript boundary. For AVO, the analogous boundary is the single Anthropic
    variation prompt plus validation-feedback retry.
- Local reference agent: `pi-mono-agent/packages/agent/src/types.ts`
  - Useful pattern: hooks such as `afterToolCall` can terminate or steer after a completed tool
    batch without corrupting transcript artifacts. For AVO, provider failures stop the loop while
    schema/validation failures stay inside the planner retry path.
- Exa search result: `Robust and Efficient Tool Orchestration via Layered Execution Structures
  with Reflective Correction`, `https://arxiv.org/html/2602.18968v1`
  - Useful fact: schema-gated tool execution benefits from localized, error-driven repair with a
    bounded budget, rather than letting invalid calls pollute the entire trajectory.
- Exa search result: `SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents`,
  `https://arxiv.org/abs/2601.16746v2`
  - Useful fact: coding-agent context pruning should retain task-relevant implementation details
    instead of compressing blindly.

Change:

- Runtime `avo/agent.py` now budgets the invalid-decision retry prompt as well as the initial
  variation prompt.
- When a planner response fails local validation, the retry preserves:
  - the static prompt head,
  - the newest prompt tail, including pending-transform or repair context,
  - the validation error and correction hint at the end.
- README now documents that validation-feedback retries use the same prompt budget.

Why:

- The previous checkpoint bounded the first Anthropic prompt, but `_decision_kwargs_with_feedback`
  appended validation feedback afterward. In long runs, a prompt already near the budget could grow
  past the intended context cap exactly when the planner needed the most precise correction signal.
- This keeps invalid tool-output recovery localized and bounded, matching the broader agent design:
  schema validation catches bad decisions, then the next planner call receives a compact, actionable
  correction without losing the newest executable context.

Verification:

- Focused retry-budget tests:
  - `.venv/bin/python -m pytest tests/test_agent.py::test_decision_feedback_retry_preserves_budget_and_latest_context tests/test_agent.py::test_valid_decision_request_retries_invalid_decision_with_feedback -q`:
    passed, 2 tests.
- Affected agent suite:
  - `.venv/bin/python -m pytest tests/test_agent.py -q`: passed, 201 tests.
- Hygiene:
  - `.venv/bin/ruff check avo tests`: passed.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 426 tests.

Decision:

- Keep the retry compaction deterministic and local to the prompt builder. Do not introduce a
  second reviewer model or learned context pruner until a concrete Anthropic trace shows the
  deterministic section/tail strategy is insufficient.

## 2026-05-10 - Checkpoint 5.51: Tighten empty score correctness reporting

Sources checked:

- NVIDIA CUDA Runtime API event documentation,
  `https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html`
  - Useful fact: CUDA events are the right GPU-side timing primitive for elapsed milliseconds, with
    completion/synchronization caveats. This matches AVO's worker-side event timing plus explicit
    synchronization path.
- PyTorch `scaled_dot_product_attention` documentation,
  `https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html`
  - Useful fact: SDPA computes query/key/value attention with optional causal masking, and dropout is
    disabled only by passing `dropout_p=0.0`. This matches AVO's correctness reference call.

Change:

- Runtime `avo/benchmark.py` now reports `all_correct=false` for an empty case set.
- Added `tests/test_benchmark_math.py::test_score_summary_treats_empty_case_set_as_not_correct`.

Why:

- The lineage gate already rejects empty case sets, and normal CLI parsing requires at least one
  sequence length. Still, the score summary itself was overstating correctness for `scores=[]`
  because Python's `all([])` is true.
- This keeps the scoring payload honest at its source: zero measured cases means no correctness
  evidence, not a vacuous correctness pass.

Verification:

- Focused score/gate/isolation suites:
  - `.venv/bin/python -m pytest tests/test_benchmark_math.py tests/test_candidate_backend.py tests/test_lineage.py tests/test_isolation.py -q`:
    passed, 35 tests.
- Live isolated A6000 score smoke:
  - `timeout 300 .venv/bin/python -m avo score --backend torch-sdpa --seq-lens 4096 --causal false --warmup 1 --repeats 1 --trials 2 --timeout-s 240`:
    passed through the parent/worker boundary with `all_correct=true`, A6000/sm86 metadata,
    two CUDA-event timing samples, median 10.068 ms, and 109.208 TFLOPS for the single
    non-causal BF16 seq4096 case.
- Hygiene:
  - `.venv/bin/ruff check avo tests`: passed.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `.venv/bin/python -m pytest -q`: passed, 427 tests.

Decision:

- Keep empty-case rejection in both places: the score payload should not claim correctness without
  cases, and the lineage gate should still reject malformed or empty score payloads defensively.

## 2026-05-10 - Checkpoint 5.52: Prefer native batch transform steps

Sources checked:

- Anthropic tool-use documentation,
  `https://docs.anthropic.com/en/docs/build-with-claude/tool-use/`
  - Useful fact: client tools are defined with an `input_schema`, and strict tool use can make tool
    inputs follow that schema.
- Anthropic define-tools documentation,
  `https://docs.anthropic.com/en/docs/build-with-claude/tool-use/implement-tool-use`
  - Useful fact: examples and descriptions matter for complex tools with nested objects or
    format-sensitive fields.

Change:

- Runtime `avo/agent.py` now exposes native `candidate_transform.steps` in the strict Anthropic
  tool schema for `op=batch`.
- `steps_json` remains accepted as a legacy fallback for old attempt records and plain JSON
  fallback responses, but the prompt and repo context now tell the planner to use a native `steps`
  array.
- Runtime README documents the preferred native `steps` array and the legacy fallback.

Why:

- The parser and materializer already supported native batch `steps`, but the tool schema only
  advertised `steps_json`. That made the planner put structured edit steps inside a JSON string,
  which is a brittle interface for the exact problem we are trying to solve: reliable,
  executable, reviewable semantic transforms.
- This is not a broader transform language and not a new CUDA guard. It makes the existing
  structured-transform interface easier for the Anthropic tool call to satisfy.

Verification:

- Focused interface tests:
  - `.venv/bin/python -m pytest tests/test_agent.py::test_decision_tool_uses_strict_schema tests/test_agent.py::test_parse_variation_decision_accepts_transform_batch tests/test_agent.py::test_parse_variation_decision_applies_batch_default_path tests/test_agent.py::test_build_repo_context_lists_local_candidates tests/test_agent.py::test_build_variation_prompt_includes_repo_context -q`:
    passed, 5 tests.
- Agent suite:
  - `.venv/bin/python -m pytest tests/test_agent.py -q`: passed, 201 tests.
- Affected agent/CLI/evolve suites:
  - `.venv/bin/python -m pytest tests/test_agent.py tests/test_cli.py tests/test_evolve.py -q`:
    passed, 343 tests.
- Hygiene:
  - `.venv/bin/ruff check avo tests`: passed.
  - `git diff --check`: passed in the runtime repo.

Decision:

- Prefer native nested JSON for semantic batch transforms. Keep `steps_json` only for backward
  compatibility and fallback response parsing.

## 2026-05-10 - Checkpoint 5.53: Align transform batch retrieval knowledge

Change:

- Runtime `knowledge/ampere.md` now says batch `candidate_transform` calls should prefer native
  `steps` arrays, with `steps_json` only as a legacy fallback.
- Runtime `knowledge/retrieval_claims.md` has a retrieval claim for the preferred native batch
  transform shape.
- Runtime `tests/test_knowledge.py` now checks that the transform-interface query retrieves the
  current native `steps` guidance and does not retrieve the old stale wording.

Why:

- After the strict Anthropic tool schema was updated, the runtime knowledge base still contained
  older guidance saying the provider-facing batch interface advertised compact `steps_json`.
  That contradicted the agent interface and could steer retrieved planner context back toward the
  brittle JSON-in-a-string path.
- This is a retrieval hygiene fix, not a new CUDA guard. The useful information is the current
  interface invariant: batch transforms should be structured, native, reviewable, and recoverable;
  the legacy string fallback exists for old traces and compatibility.

Verification:

- Focused retrieval query:
  - `uv run python -m avo knowledge-search knowledge/ampere.md --query "candidate_transform batch native steps array steps_json legacy fallback" --max-chunks 4 --max-chars 6000`:
    returned the new transform-interface claim and updated Ampere note.
- Knowledge suite:
  - `uv run pytest tests/test_knowledge.py -q`: passed, 43 tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 429 tests.

Decision:

- Keep the historical nested-schema caveat in the knowledge base, but frame it as a provider
  compatibility failure if it recurs, not as evidence that CUDA search or native semantic
  transforms are wrong.

## 2026-05-10 - Checkpoint 5.54: Capture runtime-declared candidate sources

Sources checked:

- Python data model documentation,
  `https://docs.python.org/3/reference/datamodel.html`
  - Useful fact: module `__file__` is the loaded file path when available, so using a candidate
    module's own `__file__` to report sibling runtime sources is a standard Python mechanism.
- Python `os` documentation,
  `https://docs.python.org/3/library/os.html`
  - Useful fact: `os.PathLike` and `os.fspath` are the standard interface for accepting both string
    paths and path objects.

Change:

- Runtime `avo/benchmark.py` now records `candidate_source_files` in candidate score summaries when
  a loaded candidate module exposes `AVO_SOURCE_FILES` or `__avo_source_files__`.
- Runtime `avo/evolve.py` includes those declared source paths in accepted lineage snapshots after
  normalizing them under `candidates/`.
- Runtime README and knowledge claims now document the explicit runtime source manifest hook.

Why:

- Static AST discovery already captures direct local Python imports and static extension source
  lists, but it cannot see sources assembled only when candidate code executes.
- The narrow fix is an explicit manifest hook rather than a broad import tracer. It is reviewable,
  candidate-controlled, and still constrained by the existing `candidates/` source-snapshot policy.

Verification:

- Focused behavior tests:
  - `uv run pytest tests/test_candidate_backend.py::test_candidate_backend_reports_declared_runtime_source_files tests/test_evolve.py::test_finalize_attempt_snapshots_declared_runtime_source_files tests/test_knowledge.py::test_real_ampere_corpus_retrieves_useful_claims -q`:
    passed, 36 tests.
- Affected suites:
  - `uv run pytest tests/test_candidate_backend.py tests/test_evolve.py tests/test_knowledge.py -q`:
    passed, 151 tests.
- Retrieval query:
  - `uv run python -m avo knowledge-search knowledge/ampere.md --query "candidate runtime source manifest AVO_SOURCE_FILES __avo_source_files__ lineage snapshot" --max-chunks 4 --max-chars 6000`:
    returned the new runtime-source-manifest claim.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 432 tests.

Decision:

- Keep automatic dynamic import tracing listed as missing for now. The runtime now has a clean
  explicit reporting hook for dynamically assembled CUDA sources, but it does not attempt to trace
  every import or extension load implicitly.

## 2026-05-10 - Checkpoint 5.55: Report FA2 comparison gaps in lineage summaries

Sources checked:

- FlashAttention repository,
  `https://github.com/Dao-AILab/flash-attention/`
  - Useful facts: the repository is the official implementation of FlashAttention and
    FlashAttention-2, supports Ampere GPUs, and its benchmark context includes head dimensions
    64/128 with long sequence lengths such as 4k/8k/16k. That makes it a useful comparison lane
    for the Ampere target, but not a reasonable hard acceptance gate for early candidate search.

Change:

- Runtime lineage summaries now include a derived `baseline_comparisons` section whenever a
  candidate lane and FlashAttention-2 baseline lane share the same benchmark signature.
- Each comparison records candidate geomean TFLOPS, FA2 geomean TFLOPS, candidate-vs-baseline
  ratio, baseline-vs-candidate ratio, and the signed TFLOPS gap.
- Runtime README, Ampere knowledge, retrieval claims, and tests document and verify the new
  comparison signal.

Why:

- The optimizer already preserved candidate-vs-candidate incremental progress, but the summary did
  not expose the remaining FA2 gap as a direct planner signal.
- This keeps FA2 as a comparison target rather than a pass/fail threshold. The goal is to improve
  the feedback loop with a useful optimization target, not add another brittle guard.

Verification:

- Focused lineage suite:
  - `uv run pytest tests/test_lineage.py -q`: passed, 20 tests.
- Affected lineage/knowledge suites:
  - `uv run pytest tests/test_lineage.py tests/test_knowledge.py -q`: passed, 65 tests.
- Retrieval query:
  - `uv run python -m avo knowledge-search knowledge/ampere.md --query "lineage baseline_comparisons candidate_vs_baseline FA2 gap_tflops" --max-chunks 4 --max-chars 6000`:
    returned the new baseline comparison claim and Ampere note.
- Live lineage summary check:
  - Current best candidate lane is about `0.0873504789x` the matching FA2 baseline, with a
    remaining gap of about `-100.2328556` TFLOPS.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 433 tests.

Blocked:

- Live provider-driven optimization is still blocked by Anthropic low credit even when loading
  `../avo/.env.local`, so this checkpoint improves the runtime feedback surface but does not prove
  a new autonomous CUDA improvement run.

Decision:

- Keep candidate acceptance based on prior candidate scores for the same benchmark signature.
  Report FA2 gap explicitly so the planner has a concrete optimization target without rejecting
  useful intermediate candidates.

## 2026-05-10 - Checkpoint 5.56: Keep compile repair from cycling failed payloads

Sources checked:

- NVIDIA CUDA Programming Guide, asynchronous data copies,
  `https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html`
  - Useful fact: CUDA async-copy completion is coordinated explicitly; the relevant hard
    correctness questions are completion, synchronization, and dataflow, not a standalone ban on a
    narrow copy size.
- NVIDIA libcu++ `cuda::memcpy_async` reference,
  `https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html`
  - Useful fact: on Ampere+, `cuda::memcpy_async` may lower to `cp.async` for aligned
    global-to-shared copies, with at least 4-byte alignment required by the documented conditions.
- NVIDIA Ampere Tuning Guide,
  `https://docs.nvidia.com/cuda/ampere-tuning-guide/`
  - Useful fact: Ampere adds hardware acceleration for asynchronous global-to-shared copies and
    exposes the feature through CUDA pipeline APIs.

Change:

- Runtime repair loops now keep episode-local memory of every failed edit payload during immediate
  compile, transform-materialization, and correctness repair.
- The next repair prompt includes earlier failed edit payloads from the same repair episode.
- Runtime validation rejects a repair decision that repeats any earlier failed payload from the
  same episode, not only the immediately previous failed payload.
- Runtime README, Ampere knowledge, retrieval claims, and retrieval tests document the behavior.

Why:

- The agent should fix its own compile/correctness/materialization errors instead of cycling among
  failed edits. Before this change, repair attempt 2 could accidentally replay the original failed
  payload if repair attempt 1 was different, because the unchanged-payload check only compared with
  the immediate failed attempt.
- This is not another CUDA phrase ban. It is small episode-local state for the repair loop:
  scoped, reviewable, recoverable, and tied to the clear hypothesis that repeated failed payloads do
  not repair the compiler or correctness error.

Verification:

- Focused repair and retrieval tests:
  - `uv run pytest tests/test_cli.py::test_evolve_once_rejects_repair_that_repeats_earlier_failed_payload tests/test_cli.py::test_evolve_once_repairs_candidate_compile_failure_before_finishing tests/test_knowledge.py::test_real_ampere_corpus_retrieves_useful_claims -q`:
    passed, 38 tests.
- Retrieval query:
  - `uv run python -m avo knowledge-search knowledge/ampere.md --query "compile repair episode failed edit payload replay rejected" --max-chunks 4 --max-chars 6000`:
    returned the new repair-loop claim and Ampere note.
- Affected suites:
  - `uv run pytest tests/test_cli.py tests/test_knowledge.py -q`: passed, 89 tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 435 tests.

Decision:

- Keep failed repair edits as repair-loop memory, not persistent global bans. Future repairs may
  revisit a family with a materially different semantic transform, but exact failed payload replays
  inside the same repair episode are rejected before execution.

## 2026-05-10 - Checkpoint 5.57: Capture runtime-loaded candidate Python sources

Sources checked:

- Python importlib documentation,
  `https://docs.python.org/3/library/importlib.html`
  - Useful fact: loadable modules have import-system state such as `__spec__`, and module
    file locations are normally exposed through `__file__`/origin information.
- Python sys documentation,
  `https://docs.python.org/3/library/sys.html`
  - Useful fact: `sys.modules` maps module names to already-loaded module objects, and code that
    iterates it should work on a copy because imports can change the dictionary during iteration.
- Python import-system reference,
  `https://docs.python.org/3/reference/import.html`
  - Useful fact: `sys.modules` is the first import cache and loaded modules may expose `__file__`
    for file-backed Python modules.

Change:

- Runtime candidate scoring now scans loaded modules after candidate load/scoring and records
  Python/source files whose `__file__` resolves under the same `candidates/` tree as the scored
  candidate.
- The scan is intentionally narrow:
  - it only records source-like suffixes (`.py`, `.cpp`, `.cu`, `.cuh`, `.h`, `.hpp`);
  - it skips `__pycache__`;
  - it normalizes captured paths as `candidates/...`;
  - dynamically assembled extension source lists still need static `sources=[...]` discovery or
    the explicit `AVO_SOURCE_FILES` / `__avo_source_files__` hook.
- Runtime README, Ampere knowledge, retrieval claims, and retrieval tests now describe this as
  runtime-loaded Python module capture rather than broad dynamic tracing.

Why:

- Accepted lineage snapshots already capture direct static imports, static Torch extension source
  lists, and candidate-declared runtime manifests. They did not capture helper modules imported via
  runtime string imports such as `importlib.import_module("candidates.foo")`.
- This closes the dynamic Python import part of the accepted-artifact audit gap without building a
  broad import tracer or copying arbitrary binary modules.

Verification:

- Focused runtime import and retrieval tests:
  - `uv run pytest tests/test_candidate_backend.py::test_candidate_backend_reports_runtime_imported_candidate_sources tests/test_candidate_backend.py::test_candidate_backend_reports_declared_runtime_source_files tests/test_knowledge.py::test_real_ampere_corpus_retrieves_useful_claims -q`:
    passed, 38 tests.
- Retrieval query:
  - `uv run python -m avo knowledge-search knowledge/ampere.md --query "candidate runtime source manifest dynamic import AVO_SOURCE_FILES __avo_source_files__ lineage snapshot" --max-chunks 4 --max-chars 6000`:
    returned the updated source-snapshot claim.
- Affected suites:
  - `uv run pytest tests/test_candidate_backend.py tests/test_evolve.py tests/test_knowledge.py -q`:
    passed, 154 tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 436 tests.

Decision:

- Treat runtime-loaded Python helper modules as auditable source dependencies when they live under
  the scored candidate's `candidates/` tree. Keep dynamic extension-source tracing listed as
  missing unless candidates report those files or express them in a static `sources=[...]` form.

## 2026-05-10 - Checkpoint 5.58: Capture runtime extension load sources

Sources checked:

- PyTorch C++/CUDA extension documentation,
  `https://docs.pytorch.org/docs/2.11/cpp_extension.html`
  - Useful fact: `torch.utils.cpp_extension.load(name, sources, ...)` JIT-compiles the given
    C++/CUDA source paths into an extension, loads it into the current Python process, and accepts
    `sources` as relative or absolute paths.

Change:

- Runtime candidate scoring now wraps `torch.utils.cpp_extension.load` while loading/scoring a
  candidate and records source paths passed through the `sources` argument when they normalize
  under the same `candidates/` tree.
- The capture is best effort and narrow:
  - it observes the extension loader call instead of tracing arbitrary filesystem writes;
  - it accepts string/pathlike sources from positional or keyword `sources`;
  - it normalizes only source-like files under `candidates/`;
  - it restores the original PyTorch loader after candidate scoring.
- If extension loading fails during candidate import, the failed score summary still includes the
  observed `.cpp`/`.cu` source files so the repair loop can inspect the actual artifact that failed.
- Runtime README, Ampere knowledge, retrieval claims, and retrieval tests now include runtime
  `torch.utils.cpp_extension.load(sources=[...])` capture.

Why:

- The self-repair loop needs the real candidate source set. A candidate can dynamically JIT-load
  CUDA sources during import, and a compile failure can happen before the module exposes
  `attention`. Without recording the `sources` payload before the failure, lineage and repair would
  know only that candidate import failed, not which CUDA files belonged to the failed attempt.
- This is a small semantic move toward recoverable CUDA evolution: capture the auditable source
  dependency at the tool boundary already used by candidates, without broad tracing or another
  reactive ban list.

Verification:

- Focused runtime extension, failure, dynamic import, and retrieval tests:
  - `uv run pytest tests/test_candidate_backend.py::test_candidate_backend_reports_runtime_torch_extension_sources tests/test_candidate_backend.py::test_candidate_backend_keeps_runtime_torch_extension_sources_on_load_failure tests/test_candidate_backend.py::test_candidate_backend_reports_runtime_imported_candidate_sources tests/test_knowledge.py::test_real_ampere_corpus_retrieves_useful_claims -q`:
    passed, 39 tests.
- Affected suites:
  - `uv run pytest tests/test_candidate_backend.py tests/test_evolve.py tests/test_knowledge.py -q`:
    passed, 156 tests.
- Retrieval query:
  - `uv run python -m avo knowledge-search knowledge/ampere.md --query "candidate runtime torch cpp_extension load sources dynamic import AVO_SOURCE_FILES __avo_source_files__ lineage snapshot" --max-chunks 4 --max-chars 6000`:
    returned the updated runtime extension source-capture claim.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 438 tests.

Decision:

- Treat runtime-observed `torch.utils.cpp_extension.load(sources=[...])` paths as auditable
  candidate sources when they are under `candidates/`. Keep generated source files that are never
  imported, extension-loaded, or declared via manifest out of automatic capture for now; broad
  filesystem tracing is a larger, riskier mechanism than this checkpoint needs.

## 2026-05-10 - Checkpoint 5.59: Add coherent block transform op

Sources checked:

- AgentPatterns.ai edit-format selection,
  `https://agentpatterns.ai/tool-engineering/llm-edit-format-selection/`
  - Useful fact: search/replace blocks and structure-aware edits reduce fragile line-numbered diff
    commitments by anchoring edits to unique source content or coherent code spans.
- Comby structural search/replace,
  `https://comby.dev/`
  - Useful fact: lightweight structural search/replace is useful because it lets tools operate on
    code-shaped fragments instead of brittle raw regex or line-number patch mechanics.

Change:

- Runtime `candidate_transform` now supports `replace_block_once`.
- `replace_block_once` uses the same exact unique-match materialization safety as `replace_once`,
  but its interface and prompt wording are explicitly for coherent loop/body/helper replacement.
- Planner prompts, retry feedback, README, and Ampere knowledge now distinguish:
  - `replace_once` for single-line or small expression swaps;
  - `replace_block_once` for coherent block-level semantic moves;
  - `batch` for coordinated wrapper/kernel contract changes.
- Semantic-claim validation treats `replace_block_once` as a real replacement op when checking
  claims about reduced or relocated loads.

Why:

- The earlier transform interface solved raw CUDA diff failures, but it still nudged the planner
  toward tiny text edits. The user correction was that the loop needs small semantic moves:
  scoped, reviewable, recoverable, and tied to a hypothesis, not necessarily the smallest textual
  diff.
- This adds one block-level operation without introducing an AST parser or broad new patch format.
  It gives the agent a clearer interface for replacing a coherent current code span while keeping
  exact-match failure recovery and structural CUDA preflight unchanged.

Verification:

- Focused parser/schema/materialization tests:
  - `uv run pytest tests/test_agent.py::test_parse_variation_decision_accepts_block_transform tests/test_agent.py::test_decision_tool_uses_strict_schema tests/test_evolve.py::test_run_decision_command_materializes_block_transform_before_command -q`:
    passed, 3 tests.
- Affected suites:
  - `uv run pytest tests/test_agent.py tests/test_evolve.py tests/test_knowledge.py -q`:
    passed, 350 tests.
- Retrieval query:
  - `uv run python -m avo knowledge-search knowledge/ampere.md --query "candidate_transform replace_block_once coherent loop body helper replacement semantic transform raw diff" --max-chunks 4 --max-chars 6000`:
    returned the updated transform-interface claim.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 440 tests.

Decision:

- Keep `replace_block_once` as a content-anchored block transform, not a new raw diff channel. This
  is enough to steer the planner toward coherent source spans while preserving the current
  materialize/preflight/repair path.

## 2026-05-10 - Checkpoint 5.60: Index block transform retrieval claim

Change:

- Added a high-value retrieval claim for `replace_block_once` in the runtime knowledge claims file.
- Added a retrieval test query for the distinction between `replace_block_once`, tiny text edits,
  and raw CUDA diffs.

Why:

- Checkpoint 5.59 updated the runtime and Ampere note, but the curated claim index did not yet
  expose the new operation directly. The planner should be able to retrieve the concrete interface
  rule, not only the broader "small semantic transform" principle.

Verification:

- `uv run pytest tests/test_knowledge.py::test_real_ampere_corpus_retrieves_useful_claims -q`:
  passed, 37 tests.
- `uv run pytest tests/test_knowledge.py -q`: passed, 47 tests.
- `uv run ruff check`: passed in the runtime repo.
- `git diff --check`: passed in the runtime repo.

Decision:

- Keep this as a knowledge-indexing checkpoint only. No runtime behavior changed after the
  `replace_block_once` implementation checkpoint.

## 2026-05-10 - Checkpoint 5.61: Route score-time extension builds to compile repair

External source checked:

- PyTorch `torch.utils.cpp_extension` documentation,
  `https://docs.pytorch.org/docs/2.12/cpp_extension.html`
  - Useful fact: `torch.utils.cpp_extension.load(name, sources=[...])` emits a Ninja build file,
    compiles the given sources into a dynamic library, and loads that library into the Python
    process. A failure during this stage is build feedback for the candidate extension sources,
    not a numerical correctness result.

Change:

- Score payload errors that look like CUDA/Torch extension build failures now classify as
  `score_time_compile_failure`.
- `evolve-once` routes that class to an immediate score-time compile-repair prompt instead of the
  ordinary correctness-repair prompt.
- The repair prompt includes the failed edit payload, score error summary, and runtime-captured
  `candidate_source_files` when the score payload reports them.
- Runtime README and knowledge claims now document the routing rule.

Why:

- The loop already repaired command-level compile failures, but build failures can also surface
  inside candidate scoring when PyTorch JIT-builds an extension. Treating those as
  `all_correct=false` correctness failures misleads the agent with a prompt that says the code
  compiled and ran.
- This is a routing fix, not another CUDA phrase ban: score-time extension build output is sent to
  a compile-style repair path so the agent can repair its own build error with the relevant source
  files visible.

Verification:

- Focused regression tests:
  - `uv run pytest tests/test_evolve.py::test_score_payload_extension_build_failure_is_compile_class tests/test_cli.py::test_evolve_once_repairs_score_time_extension_build_failure_before_finishing -q`:
    passed, 2 tests.
- Affected suites:
  - `uv run pytest tests/test_cli.py tests/test_evolve.py tests/test_knowledge.py -q`:
    passed, 195 tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 444 tests.

Decision:

- Keep environment failures such as missing Ninja or missing CUDA out of repair routing, but treat
  actual candidate extension build output from `nvcc`/Ninja as repairable compile feedback.

## 2026-05-10 - Checkpoint 5.62: Add supervisor strategy-reset hints

External source checked:

- `Wink: Recovering from Misbehaviors in Coding Agents`,
  `https://www.arxiv.org/pdf/2602.17037`
  - Useful fact: production coding-agent loops benefit from trajectory observers that detect
    recurring misbehaviors, then inject targeted course-correction guidance rather than relying on
    the agent to notice loops by itself.

Change:

- Runtime attempt-history summaries now add `Strategy reset candidates` when an existing
  supervisor or semantic-family signal detects stagnation.
- The reset hint remains broad and non-prescriptive: work decomposition/query-tile ownership,
  memory layout plus vectorized K/V pipeline, register/online-softmax scheduling, or a
  bottleneck-directed no-edit diagnostic.
- The hint also names recent transform families to avoid repeating unchanged.
- README, Ampere knowledge, and the retrieval-claim index now describe the strategy-reset signal.

Why:

- The previous supervisor signal correctly detected repeated command/edit fingerprints, recurring
  failure classes, recurring semantic families, and exhaustion, but its guidance mostly said "try a
  materially different direction."
- The architecture doc calls for supervisor intervention that suggests several new candidate
  directions without prescribing a specific patch. This change moves closer to that: it gives the
  planner broad Ampere search families while still requiring the next action to be a scoped,
  source-verifiable `candidate_transform` or a diagnostic tied to a concrete bottleneck.

Verification:

- Focused supervisor tests:
  - `uv run pytest tests/test_evolve.py::test_summarize_attempt_history_flags_recurring_transform_family tests/test_evolve.py::test_summarize_attempt_history_flags_unaccepted_exhaustion tests/test_evolve.py::test_summarize_attempt_history_flags_repeated_unaccepted_attempts tests/test_evolve.py::test_summarize_attempt_history_classifies_provider_failure_without_supervisor_signal -q`:
    passed, 4 tests.
- Affected suites:
  - `uv run pytest tests/test_evolve.py tests/test_knowledge.py -q`:
    passed, 152 tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.

Decision:

- Keep the supervisor reset deterministic and broad for now. Do not add an LLM supervisor call or
  a large rule tree; the current step should improve search-loop guidance without becoming another
  reactive ban list.

## 2026-05-10 - Checkpoint 5.63: Retry invalid repair decisions with feedback

External source checked:

- AgentPatterns.ai, `Failure-Driven Iteration for Improving Agent Workflows`,
  `http://agentpatterns.ai/workflows/failure-driven-iteration/`
  - Useful fact: practical agent repair loops should pass concrete failure output back into the
    next attempt and verify the resulting fix with the same tool, rather than letting the agent
    guess from generic guidance.

Change:

- Runtime compile/materialization/correctness repair now validates the returned repair decision
  before execution.
- If that repair decision violates the repair contract, for example a no-edit repair for a failed
  executable edit or a replay of an earlier failed payload, the invalid repair is not executed.
- The same repair prompt is retried once with `Repair validation feedback` that includes the
  validation error. If the decision is still invalid, the step remains a bounded planning failure.
- README and knowledge retrieval notes now document the behavior.

Why:

- The previous repair path could ask for a repair, receive an invalid repair-specific response, and
  immediately finalize the step. That meant the agent was not really repairing its own interface
  mistake inside the repair episode.
- This keeps the interface scoped and recoverable without adding another hard phrase list or an
  unbounded retry loop.

Verification:

- Affected suites:
  - `uv run pytest tests/test_cli.py tests/test_knowledge.py -q`: passed, 95 tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 447 tests.

Decision:

- Keep the retry count fixed at one feedback retry for now. Increasing repair attempts should be a
  separate decision driven by live-loop evidence, not a default escape hatch.

## 2026-05-10 - Checkpoint 5.64: Report companion source files during scoring

External source checked:

- Pantsbuild, `Dependency inference: Pants's special sauce`,
  `https://pantsbuild.org/blog/2022/10/27/why-dependency-inference`
  - Useful fact: file-level source/dependency inference is most useful when it is precise and
    scoped; inferred metadata should reduce manual declarations without becoming arbitrary
    filesystem discovery.

Change:

- Candidate scoring now adds scoped companion source directories to `candidate_source_files`.
  For example, scoring `candidates/foo_seed.py` reports source files under `candidates/foo/`
  and `candidates/foo_seed/` when those directories exist.
- The fallback also runs on candidate load failure, so score-time compile repair prompts can still
  see companion CUDA/C++ helpers when the Python candidate fails during extension setup.
- README, Ampere knowledge, and retrieval claims now describe the boundary: companion directories,
  imports, observed `torch.utils.cpp_extension.load` sources, and explicit manifests are captured;
  arbitrary generated files outside those paths still require a manifest.

Why:

- The lineage finalizer already snapshots companion directories for accepted candidates, but the
  score payload did not expose that source provenance to repair prompts.
- This closes a practical repair-loop gap without scanning all of `candidates/` or pretending the
  runtime can detect arbitrary filesystem writes.

Verification:

- Focused suites:
  - `uv run pytest tests/test_candidate_backend.py tests/test_knowledge.py -q`: passed, 60 tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 448 tests.

Decision:

- Keep automatic discovery tied to candidate naming conventions and source suffix filtering. Wider
  discovery should require a concrete failure trace before expanding the surface.

## 2026-05-10 - Checkpoint 5.65: Add structured transcript breadcrumbs

External sources checked:

- Pi mono-agent local reference, `packages/agent/README.md`
  - Useful pattern: context transformation is a first-class pre-LLM step for pruning or compaction,
    separate from LLM-format conversion. The loop can stop after a turn to compact before another
    model call, rather than aborting a provider stream or running tool work mid-compaction.
- Exa result, `packages/coding-agent/docs/compaction.md at main - badlogic/pi-mono`,
  `https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/compaction.md`
  - Useful fact: robust compaction keeps recent messages and summarizes the older span, with
    repeated compactions anchored around the prior kept boundary.
- Exa result, `Context Compaction | Everruns`,
  `https://docs.everruns.com/advanced/compaction/`
  - Useful fact: practical compaction keeps recent tool outputs verbatim and replaces older bulky
    outputs with compact summaries or placeholders while preserving traceability.

Provider probe:

- `uv run --extra agent --extra cuda python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --env-file ../avo/.env.local --timeout-s 900 --max-steps 3 --loop-json attempts/loop_after_companion_sources_20260510T_runtime.json`
  reached Anthropic but stopped after one planner step with `planner_provider_error`: the account
  credit balance is too low for the Anthropic API. No CUDA candidate command ran.

Change:

- Runtime `compact_messages` now keeps the newest messages verbatim and replaces older messages
  with a structured `<summary>` containing:
  - compacted-message count,
  - recent-message count,
  - durable-state recovery pointers,
  - bounded role/character/excerpt breadcrumbs for the first older messages,
  - an omitted-count marker when the older span is large.
- README, Ampere knowledge, and retrieval claims now describe this as deterministic local
  compaction, not a claim that full old tool output remains active in model context.

Why:

- The old helper kept recent messages but replaced all older turns with one generic sentence. That
  was too lossy for the architecture requirement around long-running agent context management.
- The new behavior preserves enough breadcrumbs for a future harness or agent to know what kind of
  context was compacted and where to recover exact state from disk.

Verification:

- Focused suites:
  - `uv run pytest tests/test_transcript.py tests/test_knowledge.py -q`: passed, 55 tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 451 tests.

Decision:

- Keep transcript compaction deterministic for now. LLM-generated summaries can be added later if
  live long-running sessions show the breadcrumb summary is insufficient.

## 2026-05-10 - Checkpoint 5.66: Bound transcript summaries to remaining budget

Change:

- Follow-up to Checkpoint 5.65: runtime `compact_messages` now applies `max_chars` to the
  structured summary when the recent tail fits inside the requested budget.
- If the recent tail alone exceeds `max_chars`, recency wins and the summary is reduced to the
  smallest possible marker, so the newest messages still remain verbatim.
- README, Ampere knowledge, and retrieval claims now state this budget boundary explicitly.

Why:

- The structured breadcrumb summary improved traceability, but the helper still needed to treat
  `max_chars` as a real budget instead of letting the older-message summary grow independently.
- This keeps long evolve-loop transcripts scoped and recoverable without dropping the freshest
  planner, compiler, or repair context.

Provider status:

- No new provider/API call was made for this checkpoint. The last probe in Checkpoint 5.65 reached
  Anthropic but stopped with `planner_provider_error` because the account credit balance was too
  low.

Verification:

- Focused suites:
  - `uv run pytest tests/test_transcript.py tests/test_knowledge.py -q`: passed, 57 tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 453 tests.

Decision:

- Keep the recent tail as the priority. Older context is recoverable from durable state and
  breadcrumbs, while losing the newest interaction would directly weaken repair behavior.

## 2026-05-10 - Checkpoint 5.67: Keep async-copy compile errors repairable

External sources checked:

- Exa search result, NVIDIA CUDA Programming Guide, `4.10. Pipelines`,
  `https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html`
  - Useful fact: CUDA pipelines are producer/consumer stage mechanisms; useful async copies need
    acquire, submit, commit, wait, consume, and release structure rather than only a copy call.
- Exa search result, NVIDIA CCCL/libcu++ `cuda::memcpy_async`,
  `https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html`
  - Useful fact: on Ampere+ `cuda::memcpy_async` may lower to `cp.async` for aligned
    global-to-shared copies with at least 4-byte alignment. This supports treating 16-byte groups
    as the preferred throughput shape, not as the only repairable compile path.

Change:

- Runtime compile-failure classification now emits `async_copy_compile_error` when compile stderr
  mentions async-copy APIs such as `cp.async`, `__pipeline*`, or `cuda::memcpy_async`.
- `async_copy_compile_error` is not in the promotable hard-preflight map, so recurring async-copy
  API/compile failures stay as repair feedback unless a concrete structural invariant is violated.
- README, Ampere knowledge, and retrieval claims now describe this boundary.

Why:

- The prior behavior could classify an async-copy compile failure as generic `cuda_syntax_error` or
  `stale_or_undefined_symbol`. Those classes can promote to hard preflight tracks that are useful
  for real syntax/symbol patterns but too blunt for async-copy API iteration.
- This keeps hard checks for actual invariants, such as invalid pipeline lifecycle or disconnected
  helper code, while letting coherent async-copy dataflow attempts reach compile repair.

Provider status:

- No Anthropic planner call was made for this checkpoint. Anthropic live loops remain blocked by
  the low-credit error recorded in Checkpoint 5.65.

Verification:

- Focused suites:
  - `uv run pytest tests/test_evolve.py::test_async_copy_compile_errors_are_not_promoted_to_hard_preflight tests/test_evolve.py::test_update_promoted_preflight_tracks_persists_recurring_class tests/test_evolve.py::test_update_promoted_preflight_tracks_persists_mixed_recurring_classes tests/test_knowledge.py -q`: passed, 55 tests.
- Affected suites:
  - `uv run pytest tests/test_evolve.py tests/test_agent.py tests/test_knowledge.py -q`:
    passed, 358 tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 455 tests.

Decision:

- Keep async-copy granularity as guidance and async-copy compile errors as repairable feedback.
  Promote only concrete structural failures, not the broad fact that an async-copy attempt failed.

## 2026-05-10 - Checkpoint 5.68: Add directed async-copy repair guidance

External sources checked:

- Exa result, RepairAgent, `https://arxiv.org/pdf/2403.17134`
  - Useful fact: program-repair loops are stronger when they feed compilation/test feedback into
    the next repair attempt instead of only running a fixed retry loop over the same context.
- Exa result, ARCS, `https://arxiv.org/html/2504.20434v2`
  - Useful fact: budgeted synthesize-execute-repair loops can stay replayable while encoding
    execution feedback into targeted prompt repair.

Change:

- Compile-repair prompts now include one class-specific guidance line when
  `failure_class=async_copy_compile_error`.
- The guidance tells the agent to repair async-copy API/include/stage/dataflow issues and not treat
  copy granularity alone as a hard rejection or return a revert-only edit.
- README, Ampere knowledge, and retrieval claims now describe this prompt boundary.

Why:

- Checkpoint 5.67 kept async-copy compile failures out of hard preflight promotion, but the repair
  prompt only exposed the class label. That left too much room for the planner to interpret the
  class as another soft ban instead of a repair target.
- This is still a tiny semantic move: only async-copy compile repairs get the extra line, and hard
  structural validators remain unchanged.

Provider status:

- No Anthropic planner call was made for this checkpoint. Anthropic live loops remain blocked by
  the low-credit error recorded in Checkpoint 5.65.

Verification:

- Focused suites:
  - `uv run pytest tests/test_cli.py::test_compile_repair_prompt_gives_async_copy_guidance tests/test_knowledge.py -q`: passed, 54 tests.
- Affected suites:
  - `uv run pytest tests/test_cli.py tests/test_evolve.py tests/test_agent.py tests/test_knowledge.py -q`:
    passed, 405 tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 457 tests.

Decision:

- Keep repair guidance class-specific. If another failure class needs directed guidance, add it
  only after a concrete loop trace shows the generic repair prompt is too ambiguous.

## 2026-05-10 - Checkpoint 5.69: Mirror async-copy guidance for score-time builds

Change:

- Score-time compile-repair prompts now reuse the same async-copy guidance when the score error
  summary mentions `cp.async`, `__pipeline*`, or `cuda::memcpy_async`.
- The existing score-time extension-build repair test now covers this case by simulating an
  undefined `__pipeline_memcpy_async` error inside a score payload.

Why:

- Some CUDA extension build failures surface only during `avo score`, not during `avo compile`.
  Those failures carry `failure_class=score_time_compile_failure`, so the class-specific compile
  prompt from Checkpoint 5.68 would not fire.
- Mirroring the guidance based on the score error text keeps async-copy repairs consistent without
  changing failure-class promotion or structural preflight behavior.

Provider status:

- No Anthropic planner call was made for this checkpoint. Anthropic live loops remain blocked by
  the low-credit error recorded in Checkpoint 5.65.

Verification:

- Focused suites:
  - `uv run pytest tests/test_cli.py::test_evolve_once_repairs_score_time_extension_build_failure_before_finishing tests/test_cli.py::test_compile_repair_prompt_gives_async_copy_guidance tests/test_knowledge.py -q`: passed, 55 tests.
- Affected suites:
  - `uv run pytest tests/test_cli.py tests/test_evolve.py tests/test_agent.py tests/test_knowledge.py -q`:
    passed, 405 tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 457 tests.

Decision:

- Detect async-copy score-time guidance from the captured error text, not by adding a new
  score-time failure class. The score-time class still correctly describes where the failure
  surfaced.

## 2026-05-10 - Checkpoint 5.70: Add wall-clock budget to evolve-loop

External sources checked:

- Exa result, CODITECT Ralph Wiggum Autonomous Agent Guide,
  `https://docs.coditect.ai/guides/ralph-wiggum-guide`
  - Useful fact: long-running autonomous loops use max iterations, max duration, checkpoints,
    health checks, and budget-exhaustion stop criteria.
- Exa result, CODITECT `/ralph-loop`, `https://docs.coditect.ai/reference/commands/ralph-loop`
  - Useful fact: loop managers evaluate termination between iterations, including max-duration
    criteria.
- Exa result, `treygoff24/autonomous-loop`, `https://github.com/treygoff24/autonomous-loop`
  - Useful fact: Codex-style loop controllers rely on explicit gates, stop hooks, and persistent
    contracts rather than an unbounded background loop.

Change:

- `avo evolve-loop` now accepts `--max-wall-time-s`.
- The value must be positive when provided.
- The loop checks the wall-clock deadline between candidate steps, records
  `stopped_reason=max_wall_time`, and includes `max_wall_time_s` plus `elapsed_wall_time_s` in the
  loop summary JSON.
- README, Ampere knowledge, and retrieval claims now document the budget behavior and why it is
  useful for long autonomous runs.

Why:

- Longer evolve runs need both a step budget and a duration budget. A high `--max-steps` alone can
  run much longer than intended when candidate compile/score/repair steps are expensive.
- The budget is checked between steps, not inside an active compile/score/repair attempt, so
  cleanup and audit artifacts stay intact.

Provider status:

- No Anthropic planner call was made for this checkpoint. Anthropic live loops remain blocked by
  the low-credit error recorded in Checkpoint 5.65.

Verification:

- Focused suites:
  - `uv run pytest tests/test_cli.py::test_evolve_loop_stops_between_steps_after_wall_time tests/test_cli.py::test_evolve_loop_runs_until_accepted_and_records_attempts tests/test_knowledge.py -q`:
    passed, 56 tests.
- Affected suites:
  - `uv run pytest tests/test_cli.py tests/test_knowledge.py -q`: passed, 101 tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 459 tests.

Decision:

- Treat wall-clock time as a run-budget stop condition, not as active supervision policy. The
  richer supervisor still needs to classify and repair candidate failures inside each step.

## 2026-05-10 - Checkpoint 5.71: Add compiler diagnostic summaries to repair prompts

External sources checked:

- Exa result, Cycle self-refinement paper, `https://arxiv.org/pdf/2403.18746`
  - Useful fact: iterative code self-refinement improves by feeding execution results and test
    feedback back into the next repair prompt.
- Exa result, RepairAgent, `https://arxiv.org/pdf/2403.17134`
  - Useful fact: autonomous repair loops update the prompt from prior command/tool results and
    keep the loop bounded by middleware instead of relying on an unstructured retry.
- Exa result, SEIDR paper, `https://arxiv.org/pdf/2503.07693`
  - Useful fact: synthesize-execute-debug-repair systems use execution and failing-test feedback
    to repair near-miss candidates while preserving a separate ranking/selection loop.

Change:

- Immediate compile-repair prompts now include `compiler_diagnostic_summary`.
- Score-time extension-build repair prompts now include `score_build_diagnostic_summary`.
- The summaries extract likely source locations, referenced symbols, and key compiler/build lines
  from the captured stderr/stdout or score error summary.
- README, Ampere knowledge, retrieval claims, and retrieval tests now document/index this behavior.

Why:

- The agent should repair its own compile errors from concrete execution feedback. A failure class
  alone is too coarse, while another hard preflight would make the loop stricter instead of more
  capable.
- This is deliberately not a new ban list. It preserves the existing bounded repair loop and gives
  the next candidate edit better evidence about where and why the build failed.

Provider status:

- No Anthropic planner call was made for this checkpoint. Anthropic live loops remain blocked by
  the low-credit error recorded in Checkpoint 5.65.

Verification:

- Focused suites:
  - `uv run pytest tests/test_cli.py::test_evolve_once_repairs_candidate_compile_failure_before_finishing tests/test_cli.py::test_evolve_once_repairs_score_time_extension_build_failure_before_finishing tests/test_cli.py::test_compile_repair_prompt_gives_async_copy_guidance tests/test_knowledge.py -q`:
    passed, 58 tests.
- Affected suites:
  - `uv run pytest tests/test_cli.py tests/test_knowledge.py -q`: passed, 102 tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 460 tests.

Decision:

- Feed compiler evidence into repair prompts. Do not promote these extracted diagnostics to hard
  preflight unless a separate recurring structural class proves it should be blocked before
  execution.

## 2026-05-10 - Checkpoint 5.72: Include candidate module path in score-time repairs

External sources checked:

- Exa result, ARISE, `https://arxiv.org/html/2605.03117v1`
  - Useful fact: repair/fault-localization agents benefit from explicit file-location records and
    structured repair outputs rather than only broad task descriptions.
- Exa result, RepairAgent, `https://arxiv.org/pdf/2403.17134`
  - Useful fact: repair loops should let agents gather and use relevant code context; compiler/test
    feedback alone is weaker if the prompt omits likely edit locations.
- Exa result, Agentic Harness for Real-World Compilers,
  `https://arxiv.org/html/2603.20075v1`
  - Useful fact: compiler-repair harnesses pass concrete reproducer/component context and validation
    feedback into the next synthesis step.

Change:

- Score-time compile-repair source context now includes the scored `candidate_path` before any
  runtime-captured `candidate_source_files`.
- The list is de-duplicated and still bounded to the first 20 entries.
- README, Ampere knowledge, retrieval claims, and retrieval tests now describe the `candidate_path`
  plus helper-source behavior.
- Removed the stale README missing-work bullet claiming source capture for companion directories,
  imports, extension sources, and source manifests was absent; that behavior is implemented and
  covered by runtime tests.

Why:

- A Torch extension build can fail even when helper-source capture is partial or empty. The repair
  prompt should still point the agent at the candidate wrapper module that initiated scoring.
- This improves source grounding for self-repair without widening command execution, adding a new
  preflight, or changing lineage acceptance.

Provider status:

- No Anthropic planner call was made for this checkpoint. Anthropic live loops remain blocked by
  the low-credit error recorded in Checkpoint 5.65.

Verification:

- Focused suites:
  - `uv run pytest tests/test_cli.py::test_evolve_once_repairs_score_time_extension_build_failure_before_finishing tests/test_knowledge.py -q`:
    passed, 56 tests.
- Affected suites:
  - `uv run pytest tests/test_cli.py tests/test_knowledge.py -q`: passed, 102 tests.
  - `uv run pytest tests/test_candidate_backend.py tests/test_evolve.py -q`: passed, 114 tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 460 tests.

Decision:

- Treat `candidate_path` as the minimum source-location context for score-time compile repairs.
  Runtime-discovered helper files remain additive evidence, not a prerequisite for repair.

## 2026-05-10 - Checkpoint 5.73: Add no-call Anthropic agent status command

External sources checked:

- Exa result, Harness CLI `harness check`,
  `https://www.mintlify.com/ayshptk/harness-cli/cli/check`
  - Useful fact: agent harnesses commonly expose a diagnostic check that reports agent/tool
    availability, versions, and API-key environment status without running a task.
- Exa result, Harness CLI agents concept,
  `https://mintlify.com/ayshptk/harness-cli/concepts/agents`
  - Useful fact: Claude/Anthropic-backed agents conventionally key off `ANTHROPIC_API_KEY`, and
    agent selection/setup should be explicit.

Change:

- Added `avo agent-status --env-file PATH`.
- The command prints the existing Anthropic status block as JSON: SDK import status, SDK version
  when available, env-file path/load status, and whether `ANTHROPIC_API_KEY` is present.
- The command does not print the secret and does not call Anthropic.
- README examples for `agent-plan`, `evolve-once`, and `evolve-loop` now pass
  `--env-file ../avo/.env.local`.
- Ampere knowledge, retrieval claims, and retrieval tests now index the no-call credential
  preflight.

Why:

- Live AVO planning depends on Anthropic credentials, but `avo env` mixes that check with CUDA and
  FA2 build diagnostics. A narrow agent-status command separates local credential/setup failures
  from provider/API failures before starting an expensive loop.
- This also makes it easier to confirm the lab env file is loaded without spending provider
  credits or leaking `ANTHROPIC_API_KEY`.

Provider status:

- `uv run python -m avo agent-status --env-file /home/ubuntu/avo/.env.local` reported
  `anthropic_api_key_present=true` and `env_file_loaded=true` without printing the secret.
- No Anthropic planner call was made for this checkpoint. Anthropic live loops remain blocked by
  the low-credit error recorded in Checkpoint 5.65 until provider quota is restored.

Verification:

- Focused suites:
  - `uv run pytest tests/test_cli.py::test_agent_status_command_prints_json_without_secret tests/test_cli.py::test_agent_status_loads_env_file_without_printing_value tests/test_cli.py::test_agent_status_reports_missing_key_without_secret tests/test_knowledge.py -q`:
    passed, 59 tests.
- Affected suites:
  - `uv run pytest tests/test_cli.py tests/test_knowledge.py -q`: passed, 104 tests.
- Command checks:
  - `uv run python -m avo agent-status --env-file /home/ubuntu/avo/.env.local`: passed and did
    not print the API key value.
  - `uv run python -m avo --help` and `uv run python -m avo agent-status --help`: exposed the new
    command and `--env-file`.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 462 tests.

Decision:

- Keep `agent-status` as a local setup preflight only. It should not call the provider, validate
  account credits, or replace the score/compile environment checks in `avo env`.

## 2026-05-11 - Checkpoint 5.74: Recheck live loop after async-copy softening

External sources checked:

- Exa result, NVIDIA CUDA Programming Guide asynchronous data copies,
  `https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html`
  - Useful fact: Ampere async copies are global-to-shared data movement mechanisms intended to
    overlap copy with compute; stage lifecycle and completion semantics are the structural issues.
- Exa result, NVIDIA CCCL/libcu++ `cuda::memcpy_async`,
  `https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html`
  - Useful fact: on Ampere+, `cuda::memcpy_async` may lower to `cp.async` for aligned
    global-to-shared copies with at least 4-byte alignment, while 16-byte aligned groups remain a
    better throughput-oriented shape.
- Exa result, NVIDIA Ampere Tuning Guide,
  `https://docs.nvidia.com/cuda/ampere-tuning-guide/`
  - Useful fact: Ampere adds hardware acceleration for async global-to-shared copies and retains
    normal CUDA programming-model guidance; this supports treating copy width as performance
    guidance unless it violates a concrete dataflow invariant.

State checked:

- Runtime `avo/agent.py` already treats narrow async-copy copy widths as
  `async_copy_granularity_preference` advisories, not hard structural preflight failures.
- Existing tests cover that scalar BF16 async-copy patches pass structural preflight and record an
  advisory.
- Runtime `README.md` already says async-copy compile failures stay repairable feedback and are not
  promotable hard preflight solely by recurrence.

Live loop recheck:

- `uv run python -m avo agent-status --env-file /home/ubuntu/avo/.env.local`: passed with
  `anthropic_api_key_present=true`, `anthropic_installed=true`, and `env_file_loaded=true`, without
  printing the secret.
- `timeout 21600 uv run python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --cwd . --env-file ../avo/.env.local --timeout-s 9600 --max-steps 12 --max-wall-time-s 14400 --compile-repair-attempts 4 --attempts-dir ./attempts --attempt-limit 128 --loop-json attempts/loop_provider_recheck_20260511T0000Z.json`:
  stopped after one planner step with `stopped_reason=planner_provider_error`.
- Anthropic returned HTTP 400 low-credit provider status before producing any candidate transform.

Decision:

- Do not make a CUDA-side or preflight-side code change from this run. The async-copy granularity
  rule is already soft; only concrete structural errors such as invalid pipeline stage lifecycle,
  disconnected helper code, malformed transform materialization, or invalid WMMA contracts remain
  hard preflight boundaries.
- Live autonomous search remains blocked on Anthropic account credits, not on the local
  `candidate_transform`/compile/repair loop.

## 2026-05-11 - Checkpoint 5.75: Expose lineage baseline comparison summary

External sources checked:

- Exa result, NVIDIA NVBench repository,
  `https://github.com/NVIDIA/nvbench/`
  - Useful fact: GPU benchmark tooling reports timing samples plus throughput/statistical summaries,
    not only a single headline time.
- Exa result, PyTorch distributed benchmark README,
  `https://github.com/pytorch/pytorch/blob/main/benchmarks/distributed/ddp/README.md`
  - Useful fact: benchmark workflows commonly emit JSON reports that preserve configuration and
    environment details so A/B comparisons can be reproduced and diffed.

Change:

- Added `avo lineage-summary PATH`.
- The command prints the existing `lineage_score_summary()` JSON that the agent already uses
  internally, including latest score, benchmark lanes, and derived `baseline_comparisons`.
- Documented the command in the runtime README near the FA2 baseline instructions.

Why:

- The current accepted target-shape candidate is correct but still far below FlashAttention-2.
  The comparison data existed inside the runtime library but was not exposed as a direct audit
  command.
- Exposing the summary keeps the FA2 gap visible without changing the lineage acceptance gate:
  candidate commits can still improve against prior candidate scores while the baseline lane records
  how far the run remains from the target.

Measurement:

- `uv run --extra cuda python -m avo score --backend candidate --candidate candidates/cuda_mma_attention_seed.py --seq-lens 4096,8192,16384,32768 --total-tokens 32768 --num-heads 16 --head-dim 128 --dtype bf16 --causal both --repeats 1 --warmup 1 --trials 3 --timeout-s 1800`:
  passed correctness and reported `geomean_tflops=9.357225004907734`.
- `uv run python -m avo lineage-summary ./lineage` now reports the committed shared-signature FA2
  comparison:
  - best committed candidate geomean: `9.593373728960215` TFLOPS;
  - FA2 baseline geomean: `109.82622931666803` TFLOPS;
  - candidate-vs-baseline ratio: `0.08735047892156171`;
  - gap: `-100.23285558770782` TFLOPS.

Verification:

- Focused:
  - `uv run pytest tests/test_cli.py::test_lineage_summary_command_prints_json tests/test_lineage.py::test_lineage_summary_keeps_baseline_and_candidate_lanes_for_same_signature -q`:
    passed, 2 tests.
- Affected:
  - `uv run pytest tests/test_cli.py tests/test_lineage.py -q`: passed, 69 tests.
- Command checks:
  - `uv run python -m avo lineage-summary ./lineage`: passed and printed `baseline_comparisons`.
  - `uv run python -m avo --help`: includes `lineage-summary`.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 463 tests.

Decision:

- Keep the new command as an audit surface only. It does not alter acceptance, scoring, baseline
  seeding, or agent planning.

## 2026-05-11 - Checkpoint 5.76: Verify candidate hard-crash containment

External sources checked:

- Exa result, PyInstaller isolation subprocess crash test,
  `https://github.com/pyinstaller/pyinstaller/blob/243d0279/tests/unit/test_isolation.py`
  - Useful fact: isolation layers should test child-process death explicitly, not only ordinary
    Python exceptions.
- Exa result, CPython ProcessPoolExecutor crash diagnostics issue,
  `https://github.com/python/cpython/issues/139462`
  - Useful fact: when a worker process terminates abruptly, preserving the child exit code is useful
    crash evidence for the parent/supervisor.
- Exa result, pytest-subprocess docs,
  `https://pytest-subprocess.readthedocs.io/en/stable`
  - Useful fact: subprocess behavior is often faked in unit tests, but here an actual child-process
    exit test is the stronger regression because the AVO risk is a real worker crash.

Change:

- Added a score-command regression test where a candidate module exits the worker process with
  code `139` during import.
- Tightened the runtime README wording:
  - Python exceptions and import failures become failed score records.
  - Hard worker exits such as segfaults or process aborts are contained as failed isolated command
    records with the child return code, instead of crashing the orchestrator.

Why:

- `tests/test_isolation.py` already covered a generic child-process crash, but not the actual
  `avo score --backend candidate` path. The architecture requirement is specifically that bad
  candidate kernels/candidate modules cannot take down the orchestrator.
- The previous README wording implied all crashes become failed score payloads. That is true for
  Python exceptions inside `score_backend`, but not for a hard process exit before the worker prints
  `AVO_RESULT_JSON`. The parent still contains it correctly, but as an isolated command failure.

Verification:

- Focused:
  - `uv run pytest tests/test_cli.py::test_score_command_contains_hard_candidate_crash tests/test_isolation.py -q`:
    passed, 4 tests.
- Affected:
  - `uv run pytest tests/test_cli.py tests/test_isolation.py -q`: passed, 53 tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 464 tests.

Decision:

- No runtime code change was needed. The implementation already contains hard worker exits through
  `run_json_worker`; the checkpoint adds coverage on the real score path and corrects the documented
  failure contract.

## 2026-05-11 - Checkpoint 5.77: Classify hard worker crashes in attempt memory

External source checked:

- Exa result, Python subprocess documentation,
  `https://docs.python.org/3/library/subprocess.html`
  - Useful fact: `subprocess` reports POSIX signal deaths as negative return codes when the direct
    child is observed; shell-mediated exits may surface signal deaths as `128 + signal`.

Change:

- Classified hard score/worker child exits as `worker_crash` in attempt summaries before they fall
  through to generic command failure.
- Added a summary regression for return code `139` so the recent-attempt memory exposes
  `class=worker_crash` and preserves the concrete return code.
- Added a recurrence regression proving repeated `worker_crash` attempts do not write a promoted
  hard preflight track.
- Updated the runtime README to document that hard worker exits are planner/supervisor feedback, not
  automatic hard preflight material.

Why:

- A hard child-process exit is materially different from an ordinary command failure. It should give
  the planner and supervisor useful repair context without turning every crash recurrence into
  another static ban.
- This keeps the failure loop pointed toward self-repair: classify the failed attempt, feed the class
  back into planning, and reserve hard preflight promotion for concrete structural invariants.

Verification:

- Focused:
  - `uv run pytest tests/test_evolve.py::test_summarize_attempt_history_classifies_worker_crash tests/test_evolve.py::test_summarize_attempt_history_reports_recent_steps -q`:
    passed, 2 tests.
  - `uv run pytest tests/test_evolve.py::test_summarize_attempt_history_classifies_worker_crash tests/test_evolve.py::test_worker_crashes_are_not_promoted_to_hard_preflight tests/test_evolve.py::test_async_copy_compile_errors_are_not_promoted_to_hard_preflight -q`:
    passed, 3 tests.
- Affected:
  - `uv run pytest tests/test_evolve.py -q`: passed, 106 tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 466 tests.

Decision:

- Keep `worker_crash` non-promotable. It is a high-signal attempt-memory class, not a sufficient
  structural preflight rule by itself.

## 2026-05-11 - Checkpoint 5.78: Repair edit-caused worker crashes immediately

Sources checked:

- Local `paper.md`: AVO should be a self-directed agent loop that uses execution feedback to
  propose, repair, critique, and verify implementation edits.
- Local `arch.md`: failed candidates stay in agent working memory, and the Level 1 loop is
  edit-compile-eval-diagnose with the agent adapting based on compiler output, profiler feedback,
  and correctness diagnostics.
- Exa result, SWE-agent NeurIPS paper,
  `https://papers.nips.cc/paper_files/paper/2024/file/5a7c947568c1b1328ccc5230172e1e7c-Paper-Conference.pdf`
  - Useful fact: agent-computer interface design matters for software agents; informative prompts,
    error messages, and test/program execution feedback are part of making agents able to revise
    code instead of relying on a single generation.

Change:

- Added `attempt_has_repairable_worker_crash` so a hard worker exit tied to an applied edit is
  treated as a repairable edit failure.
- Routed repairable worker crashes into the bounded immediate repair loop after cleanup reverts the
  failed edit.
- Added a worker-crash repair prompt that includes the failed command, return code, stdout/stderr
  crash evidence, and failed edit payload, while requiring a revised executable edit instead of a
  no-edit retry.
- Documented in the runtime README that isolated worker crashes from applied edits enter immediate
  repair.

Why:

- The previous checkpoint made `worker_crash` visible in attempt memory but still left crash repair
  to a later planner cycle. That was only a partial answer to the requirement that agents fix their
  own failed attempts.
- This keeps the class non-promotable as a hard preflight, but still lets the active variation step
  perform the AVO-style repair loop while the crash context is fresh.

Verification:

- Focused:
  - `uv run pytest tests/test_evolve.py::test_worker_crash_with_edit_is_repairable tests/test_evolve.py::test_worker_crash_without_edit_is_not_repairable tests/test_evolve.py::test_summarize_attempt_history_classifies_worker_crash -q`:
    passed, 3 tests.
  - `uv run pytest tests/test_cli.py::test_evolve_once_repairs_candidate_worker_crash_before_finishing -q`:
    passed, 1 test.
- Affected:
  - `uv run pytest tests/test_evolve.py tests/test_cli.py -q`: passed, 159 tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 469 tests.

Decision:

- A hard worker crash caused by an applied edit is repairable once inside the bounded edit-repair
  loop. Repeated `worker_crash` still must not promote to a hard preflight track by recurrence alone.

## 2026-05-11 - Checkpoint 5.79: Keep baseline-seeded lineage on target benchmark signature

Sources checked:

- Local `arch.md`: the scoring function is the target vector over sequence lengths
  `{4096, 8192, 16384, 32768}` with fixed total tokens `32768`, BF16 head dimension 128, and both
  causal modes.
- Exa result, FlashAttention-2 ICLR 2024 abstract,
  `https://proceedings.iclr.cc/paper_files/paper/2024/hash/98ed250b203d1ac6b24bbcf263e3d4a7-Abstract-Conference.html`
  - Useful fact: FlashAttention-2 is explicitly motivated by long-sequence attention bottlenecks and
    improves work partitioning to reduce low occupancy and shared-memory traffic.
- Exa result, FlashAttention-2 PDF,
  `https://tridao.me/publications/flash2/flash2.pdf`
  - Useful fact: the FA2 paper benchmarks long sequence attention with head dimensions 64/128 and
    controls total token count by adjusting batch size. This supports treating the AVO target suite
    as the committed progress signature rather than letting toy smoke shapes become progress lanes.

Change:

- Tightened `commit_score`: after a FlashAttention baseline is seeded, a candidate score must match
  the baseline benchmark case signature to be accepted into lineage.
- Smoke or shape-diagnostic scores can still run and provide attempt-memory feedback, but they now
  reject at the gate instead of creating a new accepted progress lane once the baseline exists.
- Classified this rejection in attempt summaries as `benchmark_signature_mismatch` instead of a
  generic throughput regression.
- Updated planner repo context and README wording so the agent sees the same contract: smoke scores
  are repair/correctness feedback, not evolutionary progress after baseline seeding.

Why:

- Earlier shape-graduation lanes were useful while the candidate learned to cover larger shapes, but
  the runtime now has an FA2 baseline and accepted source coverage through `seq32768`.
- Continuing to let edited candidates establish new smoke-only lanes would pull the search back
  toward unrealistic workloads and make the committed lineage look like progress when it is only
  auxiliary validation.

Verification:

- Focused:
  - `uv run pytest tests/test_lineage.py::test_baseline_rejects_candidate_with_different_benchmark_signature tests/test_lineage.py::test_baseline_does_not_block_candidate_lineage_progress tests/test_lineage.py::test_commit_score_accepts_new_benchmark_shape_lane -q`:
    passed, 3 tests.
  - `uv run pytest tests/test_evolve.py::test_summarize_attempt_history_classifies_benchmark_signature_mismatch -q`:
    passed, 1 test.
  - `uv run pytest tests/test_agent.py::test_build_repo_context_lists_local_candidates -q`:
    passed, 1 test.
- Affected:
  - `uv run pytest tests/test_lineage.py tests/test_evolve.py tests/test_agent.py tests/test_cli.py -q`:
    passed, 383 tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 471 tests.

Decision:

- Keep smoke scores executable because they are still useful diagnostics, especially for repairs.
  The committed candidate lineage, once FA2 is seeded, must stay on the target benchmark signature.

## 2026-05-11 - Checkpoint 5.80: Always ground planner context in broad CUDA knowledge

Sources checked:

- Exa result, NVIDIA CUDA C++ Programming Guide, Pipelines,
  `https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html`
  - Useful fact: CUDA pipelines coordinate staged producer/consumer work; wait-prior behavior is
    tied to committed stages, so wait/commit ordering is a real lifecycle invariant rather than a
    stylistic preference.
- Exa result, NVIDIA Ampere Tuning Guide,
  `https://docs.nvidia.com/cuda/ampere-tuning-guide/`
  - Useful fact: Ampere exposes hardware-accelerated global-to-shared async copies through CUDA
    pipeline APIs, and sm86 has 100 KB shared memory per SM plus explicit sm86 compilation benefits.

Attempted live loop:

- Ran:
  `uv run python -m avo evolve-loop --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --env-file ../avo/.env.local --max-steps 4 --max-wall-time-s 3600 --timeout-s 240 --compile-repair-attempts 2 --loop-json attempts/latest-loop.json`
- Result: stopped after one planning step with `planner_provider_error` because Anthropic returned
  low credit balance before producing a candidate. No CUDA candidate command ran.
- Current full-suite lineage gap before the blocked loop: candidate geometric mean about
  9.59 TFLOPS versus FlashAttention-2 about 109.83 TFLOPS on the target signature.

Change:

- Updated runtime planner context assembly so broad CUDA grounding is guaranteed when the dynamic
  retriever query misses it.
- The planner now appends bounded fallback chunks from both `knowledge/b/cuda_general.md` and
  `knowledge/b/cuda_programming_practice.md`, checked against the original retrieved context so one
  supplement cannot accidentally suppress the other.
- Updated README to document the fallback behavior.

Why:

- The user asked for general CUDA knowledge and for the agent to reason about CUDA practice, not just
  memorize local failure phrases. The broad files contain execution model, memory hierarchy,
  synchronization, profiling, and semantic-transform workflow grounding; they should reliably reach
  the planner even when the current attempt-history query is dominated by local Ampere failures.

Verification:

- Focused:
  - `uv run pytest tests/test_cli.py::test_planning_context_includes_general_cuda_context tests/test_cli.py::test_general_cuda_context_supplements_missing_broad_files tests/test_knowledge.py -q`:
    passed, 58 tests.
- Affected:
  - `uv run pytest tests/test_cli.py tests/test_knowledge.py -q`: passed, 108 tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in both runtime and lab repos.
- Full runtime suite:
  - `uv run pytest -q`: passed, 472 tests.
- Retrieval smoke:
  - `uv run python -m avo knowledge-search knowledge/ampere.md --query "General CUDA Working Knowledge Execution Model Memory Spaces CUDA Kernel Design Practice semantic transformation" --max-chunks 6 --max-chars 10000`
    retrieved chunks from both `b/cuda_general.md` and `b/cuda_programming_practice.md`.

Decision:

- Keep provider/API failures as stop conditions for the live Anthropic loop. They are external
  execution blockers, not CUDA search evidence, so repeating the loop without credits cannot improve
  the kernel.

## 2026-05-11 - Checkpoint 5.81: Make TFLOP accounting auditable in score metadata

Sources checked:

- Exa fetch, FlashAttention benchmark source,
  `https://github.com/Dao-AILab/flash-attention/blob/3387de49/benchmarks/benchmark_flash_attention.py`
  - Useful fact: the upstream benchmark helper counts forward attention as
    `4 * batch * seqlen**2 * nheads * headdim`, divided by 2 for causal cases.
- Exa fetch, FlashAttention paper,
  `https://arxiv.org/pdf/2205.14135`
  - Useful fact: the paper frames exact attention performance around the QK, softmax, and PV
    computation while emphasizing IO/memory movement as the key bottleneck; TFLOP/s alone is not a
    complete performance explanation.

Change:

- Kept the runtime's existing FlashAttention-compatible FLOP convention instead of switching to an
  exact triangular causal-pair count, because the project baseline is FlashAttention-2 and lineage
  comparisons should remain convention-compatible with upstream FA benchmarking.
- Added explicit `flop_accounting` metadata to every benchmark summary:
  - name: `flash_attention_forward_compatible`
  - formula: `4 * batch_size * num_heads * seq_len**2 * head_dim // (2 if causal else 1)`
  - scope: forward QK and PV matmul FLOPs only; excludes softmax, masking, and projections
  - causal convention: `half_dense`
- Updated README scoring docs and the benchmark metadata unit test.

Why:

- Accurate TFLOP measurement is one of the core engineering requirements, but "accurate" has two
  parts here: computing consistently and making the convention visible. The previous numbers were
  consistent with the common FlashAttention convention, but the convention was implicit in code and
  absent from lineage JSON.
- Making the convention auditable avoids confusing future candidate comparisons, especially if a
  later analysis chooses to report exact causal triangular work or IO-derived metrics alongside
  FlashAttention-compatible TFLOP/s.

Verification:

- Focused:
  - `uv run pytest tests/test_benchmark_math.py -q`: passed, 7 tests.
- Affected:
  - `uv run pytest tests/test_benchmark_math.py tests/test_lineage.py tests/test_cli.py -q`:
    passed, 80 tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 472 tests.

Decision:

- Keep FA-compatible TFLOP/s as the primary score metric for baseline comparability, and rely on the
  new metadata to make that choice explicit. If the search loop starts making IO-oriented tradeoffs,
  add separate IO/traffic diagnostics rather than silently changing the headline TFLOP convention.

## 2026-05-11 - Checkpoint 5.82: Add OpenRouter Opus 4.7 planner provider

Sources checked:

- Exa fetch, OpenRouter Claude Opus 4.7 quickstart,
  `https://openrouter.ai/anthropic/claude-opus-4.7/api`
  - Useful fact: the concrete OpenRouter model slug is `anthropic/claude-opus-4.7`; OpenRouter
    presents this model as suited for long-running asynchronous agent workflows.
- Exa fetch, OpenRouter chat-completions API reference,
  `https://openrouter.ai/docs/api-reference/chat-completion?explorer=true`
  - Useful fact: OpenRouter accepts `POST https://openrouter.ai/api/v1/chat/completions` with a
    bearer `Authorization` header and OpenAI-style `messages`, `model`, and response-format fields.
- Exa fetch, OpenRouter Claude 4.7 migration guide,
  `https://openrouter.ai/docs/guides/evaluate-and-optimize/model-migrations/claude-4-7`
  - Useful fact: Claude 4.7 Opus ignores sampling parameters such as temperature/top_p, supports
    adaptive reasoning, and uses `verbosity`; `xhigh` is available for this model.

Change:

- Added a selectable planner provider path:
  - default provider remains `anthropic`
  - `--provider openrouter` or `AVO_AGENT_PROVIDER=openrouter` uses the OpenRouter chat-completions
    API
  - when OpenRouter is selected and the caller has not overridden `--model`, the planner uses
    `anthropic/claude-opus-4.7`
  - OpenRouter requests use JSON-schema response format first, then fall back to JSON-object format
    if the provider rejects the schema wrapper
  - `AVO_OPENROUTER_VERBOSITY` can override verbosity; Opus 4.7 defaults to `xhigh`
- Added `OPENROUTER_API_KEY` presence reporting to `agent-status` without printing secret values.
- Extended planner-provider error classification so OpenRouter payment/credit failures remain
  provider outages instead of becoming CUDA search failures.
- Updated README agent workflow docs with OpenRouter commands.

Why:

- Anthropic direct planning was blocked by provider credit errors. The user asked to run the long
  loop via OpenRouter/Opus 4.7. A provider switch is safer than replacing the Anthropic path because
  the architecture still says to stay in the Anthropic ecosystem, and OpenRouter is routing the same
  Anthropic model family while leaving the direct Anthropic implementation available.
- The existing parser already supports plain JSON text responses, so the minimal port is one
  provider call boundary rather than a rewrite of decision validation, repair, lineage, or transform
  materialization.

Verification:

- Focused:
  - `uv run pytest tests/test_agent.py::test_openrouter_kwargs_use_opus_47_json_schema_defaults tests/test_agent.py::test_openrouter_response_falls_back_to_json_object_on_schema_rejection tests/test_agent.py::test_request_variation_decision_uses_openrouter_provider tests/test_cli.py::test_agent_status_reports_openrouter_key_without_secret tests/test_evolve.py::test_summarize_attempt_history_classifies_provider_failure_without_supervisor_signal -q`:
    passed, 5 tests.
- Affected:
  - `uv run pytest tests/test_agent.py tests/test_cli.py tests/test_evolve.py -q`: passed, 367
    tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Full runtime suite:
  - `uv run pytest -q`: passed, 476 tests.

Decision:

- Use OpenRouter only through `OPENROUTER_API_KEY` in the process environment or an untracked env
  file; do not write user-provided keys into tracked files or logs.
- Start the requested long loop only after this provider-port checkpoint is committed and pushed, so
  any candidate accepted by the loop is based on committed orchestrator code.

## 2026-05-11 - Checkpoint 5.83: Surface OpenRouter error payloads during planning

Attempted live loop:

- Started:
  `uv run python -m avo evolve-loop --provider openrouter --model anthropic/claude-opus-4.7 --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 80 --max-wall-time-s 21600 --timeout-s 360 --compile-repair-attempts 2 --loop-json attempts/openrouter-long-loop.json`
- Result: stopped after one planning step with `planner_provider_error`.
- Failure detail: `OpenRouter response missing choices`.

Change:

- Fixed the OpenRouter response boundary to detect a top-level provider `error` payload even when the
  HTTP call itself returns parseable JSON.
- Such payloads now raise `OpenRouterAPIError` with the provider message instead of falling through
  to a misleading missing-choices validation error.
- Kept the schema-wrapper fallback path: if OpenRouter rejects the JSON-schema response format with a
  400/422-style error, the planner retries once with JSON-object response format.

Why:

- Long-loop supervision needs accurate provider/error classification. A provider rejection is not a
  malformed planner decision and should not be treated as CUDA search evidence.
- The first live OpenRouter run showed that the response handler needed to classify provider error
  JSON before trying to read `choices`.

Verification:

- Focused:
  - `uv run pytest tests/test_agent.py::test_openrouter_response_falls_back_to_json_object_on_schema_rejection tests/test_agent.py::test_openrouter_message_text_surfaces_error_payload tests/test_agent.py::test_request_variation_decision_uses_openrouter_provider -q`:
    passed, 3 tests.
- Affected:
  - `uv run pytest tests/test_agent.py tests/test_cli.py tests/test_evolve.py -q`: passed, 368
    tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.

Decision:

- Re-run the requested long OpenRouter loop after committing and pushing this provider-response
  boundary fix.

## 2026-05-11 - Checkpoint 5.84: Bound OpenRouter body reads before long loop restart

Attempted live loop:

- Started:
  `uv run python -m avo evolve-loop --provider openrouter --model anthropic/claude-opus-4.7 --lineage ./lineage --knowledge knowledge/ampere.md --attempts-dir ./attempts --max-steps 80 --max-wall-time-s 21600 --timeout-s 360 --compile-repair-attempts 2 --loop-json attempts/openrouter-long-loop-2.json`
- Result: interrupted before step 1 after the first planner call remained nonproductive for about
  twenty minutes.
- Stack trace showed OpenRouter had returned headers and the client was blocked in
  `response.read()` while reading a chunked body.

Change:

- Added a whole-request OpenRouter deadline around `urlopen()` and response body read.
- Local OpenRouter timeouts now become `OpenRouterAPIError` with status 408, which the planner
  retry loop treats as transient.
- Added a regression test for a stalled OpenRouter response body read.

Why:

- Socket timeouts alone were not enough once the provider returned headers and kept the chunked body
  open. That could consume the entire long evolve run before recording a single attempt.
- The loop should either get a planner decision or classify/retry a provider timeout; it should not
  silently park before CUDA search begins.

Verification:

- Focused:
  - `uv run pytest tests/test_agent.py::test_openrouter_response_falls_back_to_json_object_on_schema_rejection tests/test_agent.py::test_openrouter_message_text_surfaces_error_payload tests/test_agent.py::test_openrouter_chat_completion_maps_body_read_timeout tests/test_agent.py::test_request_variation_decision_uses_openrouter_provider -q`:
    passed, 4 tests.
- Affected:
  - `uv run pytest tests/test_agent.py tests/test_cli.py tests/test_evolve.py -q`: passed, 369
    tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.

Decision:

- Restart the requested long loop after this timeout-boundary fix is committed and pushed.

## 2026-05-11 - Checkpoint 5.85: Loosen planner metadata and score-profiler validation

Attempted live loop:

- Restarted OpenRouter/Opus 4.7 long loop with:
  `--max-steps 160 --max-wall-time-s 28800 --timeout-s 600 --compile-repair-attempts 3`
- Step 1 produced a real `candidate_transform`: split the MMA PV WMMA loop across two warps.
  It compiled and passed correctness, but gate rejected it because geomean regressed from
  9.5078 to 9.4353 TFLOP/s.
- Subsequent planner attempts repeatedly failed validation instead of producing executable
  transforms:
  - repeated an unmodified MMA seed score,
  - omitted nonsemantic metadata fields (`files_to_inspect`, `expected_effect`, `risk`),
  - framed an `avo score` command as if it could collect profiler metrics.

Change:

- `VariationDecision.from_mapping()` now fills missing nonsemantic metadata:
  - `files_to_inspect` is inferred from structured transform paths or raw patch paths,
  - `expected_effect` and `risk` get conservative validation-oriented defaults.
- `avo score` still rejects no-edit diagnostics that pretend to collect profiler data, but no longer
  rejects a real `candidate_transform` score only because the prose mentions profiler-style concepts.

Why:

- The loop should not discard otherwise reviewable transform payloads because OpenRouter omitted
  bookkeeping fields that can be derived safely.
- The score/profiler rule was too strict for edited candidates: score is the correct validation path
  for a transform even if the surrounding hypothesis mentions occupancy, bottlenecks, or similar
  concepts. The hard restriction should apply to no-edit score diagnostics that claim unavailable
  profiler evidence.

Verification:

- Focused:
  - `uv run pytest tests/test_agent.py::test_parse_variation_decision_defaults_missing_nonsemantic_metadata tests/test_agent.py::test_parse_variation_decision_rejects_score_claiming_profiler_metrics tests/test_agent.py::test_parse_variation_decision_allows_transform_score_with_profiler_wording tests/test_agent.py::test_decision_feedback_explains_missing_required_keys_error tests/test_agent.py::test_decision_feedback_explains_score_profiler_metric_error -q`:
    passed, 5 tests.
- Affected:
  - `uv run pytest tests/test_agent.py tests/test_cli.py tests/test_evolve.py -q`: passed, 371
    tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.

Decision:

- Restart the OpenRouter long loop after this planner-boundary fix is committed and pushed.

## 2026-05-11 - Checkpoint 5.86: Default missing descriptive planner fields

Attempted live loop:

- Restarted OpenRouter/Opus 4.7 long loop after checkpoint 5.85.
- The next completed planner step still failed validation after three retries because the provider
  omitted `hypothesis`.

Change:

- Extended the planner-boundary defaulting to cover descriptive fields:
  - missing `candidate_edit` defaults to a transform-oriented edit description when an edit payload
    exists, otherwise to a no-edit diagnostic description,
  - missing `hypothesis` defaults from `candidate_edit`, the `avo` subcommand, or a conservative
    generic AVO step.

Why:

- OpenRouter can return partially shaped JSON even when asked for a schema-shaped decision. The
  orchestrator should require executable semantics (`next_command`, edit/no-edit consistency, patch
  preflight, compile/score guards), but it should not spend whole evolve steps rejecting missing
  prose fields that can be safely reconstructed.

Verification:

- Focused:
  - `uv run pytest tests/test_agent.py::test_parse_variation_decision_defaults_missing_descriptive_fields tests/test_agent.py::test_parse_variation_decision_defaults_missing_nonsemantic_metadata tests/test_agent.py::test_parse_variation_decision_allows_transform_score_with_profiler_wording -q`:
    passed, 3 tests.
- Affected:
  - `uv run pytest tests/test_agent.py tests/test_cli.py tests/test_evolve.py -q`: passed, 372
    tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.

Decision:

- Restart the OpenRouter long loop on the pushed follow-up fix.

## 2026-05-11 - Checkpoint 5.87: Prioritize pending compile-only transform scoring

Attempted live loop:

- Restarted OpenRouter/Opus 4.7 long loop after checkpoint 5.86.
- The loop made progress but still spent planner steps on recoverable control issues:
  - a structured transform omitted its operation `path`,
  - a planner decision repeated a successful compile-only transform instead of scoring it.

Change:

- `VariationDecision` now infers a missing structured-transform `path` from a single
  `files_to_inspect` candidate source path.
- `evolve-loop` now checks for a pending successful compile-only transform before each planner step
  and scores that transform directly instead of asking the planner what to do next.

Why:

- Missing `path` is recoverable when the decision identifies exactly one candidate file.
- A successful compile-only transform is a known control obligation. The loop should score it
  immediately; routing that obligation back through a general planner creates avoidable invalid
  decisions and burns provider calls.

Verification:

- Focused:
  - `uv run pytest tests/test_agent.py::test_parse_variation_decision_infers_missing_transform_path_from_single_file tests/test_cli.py::test_evolve_loop_scores_pending_compile_transform_before_planning tests/test_cli.py::test_evolve_loop_scores_pending_compile_transform_at_step_limit -q`:
    passed, 3 tests.
- Affected:
  - `uv run pytest tests/test_agent.py tests/test_cli.py tests/test_evolve.py -q`: passed, 374
    tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.

Decision:

- Restart the long OpenRouter loop after committing and pushing this loop-control checkpoint.

## 2026-05-11 - Checkpoint 5.88: Add start/end anchored block transform

Attempted live loop:

- Restarted OpenRouter/Opus 4.7 long loop after checkpoint 5.87.
- The loop continued to spend attempts on invalid transform shapes. One failure was useful:
  the planner emitted a block edit with `find_start` and `find_end`, which expressed a coherent
  source-region replacement but was not supported by the transform API.

Change:

- Added `replace_between_once` to `candidate_transform`.
- The new op uses:
  - `find_start`: a unique inclusive start anchor,
  - `find_end`: a unique inclusive end anchor,
  - `replace`: the replacement text for that anchored region.
- Added parser/schema support, prompt/repo-context guidance, materialization support, and tests.

Why:

- This addresses the earlier interface critique directly: a scoped semantic edit can be reviewable
  and recoverable without requiring the planner to reproduce a large exact `find` block.
- Start/end anchored replacement is a small coherent transformation primitive, not a raw CUDA diff.

Verification:

- Focused:
  - `uv run pytest tests/test_agent.py::test_parse_variation_decision_accepts_replace_between_transform tests/test_agent.py::test_decision_tool_uses_strict_schema tests/test_evolve.py::test_materialize_candidate_transform_replaces_between_anchors tests/test_agent.py::test_parse_variation_decision_infers_missing_transform_path_from_single_file -q`:
    passed, 4 tests.
- Affected:
  - `uv run pytest tests/test_agent.py tests/test_cli.py tests/test_evolve.py -q`: passed, 376
    tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.

Decision:

- Restart the OpenRouter long loop after committing and pushing this transform-interface extension.

## 2026-05-11 - Checkpoint 5.89: Normalize no-edit diagnostic prose

Attempted live loop:

- Restarted OpenRouter/Opus 4.7 long loop after checkpoint 5.88.
- The next completed step failed validation because a no-edit diagnostic had `edit_mode="no_edit"`
  and no edit payload, but `candidate_edit` did not start with the exact `No edit;` prefix.

Change:

- Normalize explicit no-edit decisions with no edit payload by adding the `No edit;` prefix when
  it is missing.

Why:

- The prefix is bookkeeping for the orchestrator, not a CUDA invariant. Rejecting an otherwise
  bounded no-edit diagnostic over this wording wastes planner retries and loop steps.
- Edit/no-edit safety is still enforced by the absence of edit payload and by command/history
  validators.

Verification:

- Focused:
  - `uv run pytest tests/test_agent.py::test_parse_variation_decision_normalizes_no_edit_prefix tests/test_agent.py::test_parse_variation_decision_accepts_replace_between_transform tests/test_evolve.py::test_materialize_candidate_transform_replaces_between_anchors -q`:
    passed, 3 tests.
- Affected:
  - `uv run pytest tests/test_agent.py tests/test_cli.py tests/test_evolve.py -q`: passed, 377
    tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.

Decision:

- Restart the OpenRouter long loop after this small normalizer is committed and pushed.

## 2026-05-11 - Checkpoint 5.90: Retry planner decisions on attempt-history validation

Attempted live loop:

- Restarted OpenRouter/Opus 4.7 long loop after checkpoint 5.89.
- The loop again recorded a failed step for an attempt-history violation: the planner repeated a
  candidate transform that was already known from prior attempts.

Change:

- Added a planner-history validation retry around normal planning decisions.
- If `validate_decision_against_attempt_history()` rejects a planner decision, the error is appended
  to the planner attempt-history context and the planner is asked for a corrected decision before an
  evolve step is recorded.

Why:

- Parser/schema retries were already internal to the planner call, but history validation happened
  after the call and burned full loop steps. This was exactly the wrong failure boundary for
  repeated transform families.
- The agent should repair its own invalid planning decision when the orchestrator can state the
  history constraint concretely.

Verification:

- Focused:
  - `uv run pytest tests/test_cli.py::test_evolve_loop_retries_history_invalid_planner_decision tests/test_agent.py::test_parse_variation_decision_normalizes_no_edit_prefix tests/test_agent.py::test_parse_variation_decision_accepts_replace_between_transform tests/test_evolve.py::test_materialize_candidate_transform_replaces_between_anchors -q`:
    passed, 4 tests.
  - After formatting fix:
    `uv run pytest tests/test_cli.py::test_evolve_loop_retries_history_invalid_planner_decision -q`:
    passed, 1 test.
- Affected:
  - `uv run pytest tests/test_agent.py tests/test_cli.py tests/test_evolve.py -q`: passed, 378
    tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.

Decision:

- Restart the OpenRouter long loop after committing and pushing this planner-retry boundary fix.

## 2026-05-11 - Checkpoint 5.91: Infer transform path from compile source

Attempted live loop:

- Restarted OpenRouter/Opus 4.7 long loop after checkpoint 5.90.
- A planner response still failed parser validation because a batch transform step omitted `path`.

Change:

- Extended candidate-transform path defaulting to use `next_command --source` as a fallback when
  `files_to_inspect` does not provide exactly one candidate source path.
- This lets pathless batch steps inherit the compile source path when the planner clearly intends to
  compile that source.

Why:

- A compile command with one `--source` is an unambiguous path signal. Rejecting a pathless transform
  batch in that case is another avoidable planner-interface failure.

Verification:

- Focused:
  - `uv run pytest tests/test_agent.py::test_parse_variation_decision_infers_batch_path_from_compile_source tests/test_agent.py::test_parse_variation_decision_infers_missing_transform_path_from_single_file tests/test_agent.py::test_parse_variation_decision_accepts_replace_between_transform -q`:
    passed, 3 tests.
- Affected:
  - `uv run pytest tests/test_agent.py tests/test_cli.py tests/test_evolve.py -q`: passed, 379
    tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.

Decision:

- Restart the OpenRouter long loop after committing and pushing this path-defaulting fix.

## 2026-05-11 - Checkpoint 5.92: Normalize transform target_path alias

Attempted live loop:

- Restarted OpenRouter/Opus 4.7 long loop after checkpoint 5.91.
- The loop produced several valid compile/score candidates, then one planner response failed parser
  validation because it used `target_path` instead of `path` inside a candidate transform.

Change:

- Normalize `target_path` to `path` in candidate transforms, including nested batch steps.

Why:

- `target_path` is an obvious alias for `path`; rejecting it is an interface-shape failure, not a
  CUDA safety check.
- This keeps the transform interface scoped and structured while making it less brittle for model
  JSON variants.

Verification:

- Focused:
  - `uv run pytest tests/test_agent.py::test_parse_variation_decision_normalizes_target_path_alias tests/test_agent.py::test_parse_variation_decision_infers_batch_path_from_compile_source tests/test_agent.py::test_parse_variation_decision_accepts_replace_between_transform -q`:
    passed, 3 tests.
- Affected:
  - `uv run pytest tests/test_agent.py tests/test_cli.py tests/test_evolve.py -q`: passed, 380
    tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.

Decision:

- Restart the OpenRouter long loop after committing and pushing this alias normalizer.

## 2026-05-11 - Checkpoint 5.93: Normalize transform replacement alias

Attempted live loop:

- Restarted OpenRouter/Opus 4.7 long loop after checkpoint 5.92.
- A planner response failed parser validation because it used `replacement` instead of `replace`
  inside a candidate transform operation.

Change:

- Normalize `replacement` to `replace` in candidate transforms, including nested batch steps.

Why:

- `replacement` is an obvious spelling variant for the replacement body. Accepting it keeps the
  transform interface structured while avoiding brittle schema-shape failures.

Verification:

- Focused:
  - `uv run pytest tests/test_agent.py::test_parse_variation_decision_normalizes_replacement_alias tests/test_agent.py::test_parse_variation_decision_normalizes_target_path_alias tests/test_agent.py::test_parse_variation_decision_accepts_replace_between_transform -q`:
    passed, 3 tests.
- Affected:
  - `uv run pytest tests/test_agent.py tests/test_cli.py tests/test_evolve.py -q`: passed, 381
    tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.

Decision:

- Restart the OpenRouter long loop after committing and pushing this alias normalizer.

## 2026-05-11 - Checkpoint 5.94: Recognize global-to-shared WMMA load relocation

Attempted live loop:

- Restarted OpenRouter/Opus 4.7 long loop after checkpoint 5.93.
- The planner repeatedly proposed K/V shared-memory staging. The semantic guard rejected those
  plans because the number of WMMA load sites stayed constant, even when the source moved from a
  global pointer (`k + ...` / `v + ...`) to a shared tile (`k_tile` / `v_tile`).

Change:

- Extended the load-reduction semantic check to treat WMMA operand loads that move from global
  memory pointers to shared tile/shared buffers as source-verifiable load relocation.

Why:

- For shared staging, the WMMA instruction count can remain the same while the memory source changes
  from global to shared. Counting load sites alone was too strict and blocked a real semantic move.

Verification:

- Focused:
  - `uv run pytest tests/test_agent.py::test_parse_variation_decision_allows_v_global_to_shared_load_relocation tests/test_agent.py::test_parse_variation_decision_rejects_claimed_v_reuse_without_v_load_change tests/test_agent.py::test_parse_variation_decision_allows_q_load_hoist_out_of_named_loop -q`:
    passed, 3 tests.
  - After formatting fix:
    `uv run pytest tests/test_agent.py::test_parse_variation_decision_allows_v_global_to_shared_load_relocation -q`:
    passed, 1 test.
- Affected:
  - `uv run pytest tests/test_agent.py tests/test_cli.py tests/test_evolve.py -q`: passed, 382
    tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.

Decision:

- Restart the OpenRouter long loop after committing and pushing this semantic-check fix.

## 2026-05-11 - Checkpoint 5.95: Make OpenRouter response budget configurable

Attempted live loop:

- Restarted OpenRouter/Opus 4.7 loop 15 with 300 steps and 12-hour wall time.
- Useful behavior observed:
  - K shared-memory staging compiled, then the pending transform was scored and rejected correctly
    at 6.9662 geomean TFLOP/s versus 9.5078 best.
  - Q shared-memory staging compiled, then the pending transform was scored and rejected correctly
    at 7.6580 geomean TFLOP/s.
  - PV dual-warp chunk splitting compiled, then the pending transform was scored and rejected
    correctly at 9.3925 geomean TFLOP/s.
- The loop later stopped with an OpenRouter HTTP 402 provider error because the account had enough
  remaining credit for only a smaller completion than the hardcoded 4000-token request.

Change:

- Added `AVO_OPENROUTER_MAX_TOKENS` so the OpenRouter planner response budget can be lowered
  without changing model or editing code.
- Kept the default at 4000 tokens.

Why:

- Provider budget limits are operational, not a CUDA-search failure. The loop should be restartable
  under tighter completion caps when the account has limited remaining credit.

Verification:

- Focused:
  - `uv run pytest tests/test_agent.py::test_openrouter_kwargs_use_opus_47_json_schema_defaults tests/test_agent.py::test_openrouter_kwargs_allows_max_tokens_override tests/test_agent.py::test_openrouter_kwargs_ignores_invalid_max_tokens_override -q`:
    passed, 6 tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.

Decision:

- Commit and push the max-token knob, then restart the OpenRouter loop with a smaller response
  budget.

## 2026-05-11 - Checkpoint 5.96: Make variation prompt budget configurable

Attempted live loop:

- Restarted OpenRouter/Opus 4.7 with `AVO_OPENROUTER_MAX_TOKENS=1200`.
- OpenRouter then rejected the request because the planner prompt exceeded the account's remaining
  prompt-token ceiling: 27857 prompt tokens versus 7114 allowed.

Change:

- Added `AVO_VARIATION_PROMPT_MAX_CHARS` and wired it through initial variation-prompt rendering
  and validation-feedback retry prompts.
- The default remains `MAX_VARIATION_PROMPT_CHARS=120000`; the new knob only narrows the prompt
  when the runtime environment asks for it.

Why:

- The provider failure was caused by an operational account limit, not by a CUDA candidate. The loop
  needs a way to continue with a compact prompt when the selected provider/key has a smaller context
  budget.

Verification:

- Focused:
  - `uv run pytest tests/test_agent.py::test_request_variation_decision_uses_openrouter_provider tests/test_agent.py::test_decision_feedback_retry_uses_prompt_budget_override tests/test_agent.py::test_decision_feedback_retry_preserves_budget_and_latest_context -q`:
    passed, 3 tests.
- Affected:
  - `uv run pytest tests/test_agent.py tests/test_cli.py tests/test_evolve.py -q`: passed, 388
    tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.
- Dry sizing:
  - `AVO_VARIATION_PROMPT_MAX_CHARS=10000` produced a 9488-character variation prompt from current
    knowledge/lineage inputs.

Decision:

- Commit and push the prompt-budget knob, then retry the OpenRouter loop with compact prompt and
  smaller completion budget.

## 2026-05-11 - Checkpoint 5.97: Reject missing candidate score paths

Attempted live loop:

- After compacting the OpenRouter prompt and response budgets, the provider still had very little
  remaining credit.
- One ultra-compact planner response chose a no-edit score diagnostic for
  `candidates/baseline.py`, which does not exist. The benchmark step ran and failed with
  `FileNotFoundError`, wasting a loop step.

Change:

- Added next-command preflight for `avo score --backend candidate --candidate ...` and
  `avo profile --backend candidate --candidate ...`.
- The preflight now requires `--candidate` to reference an existing repo-relative candidate wrapper,
  or a candidate path added by the same raw non-CUDA `candidate_patch`.
- Added retry feedback that tells the planner to use an existing wrapper such as
  `candidates/cuda_mma_attention_seed.py` and not invent placeholder paths.

Why:

- Missing candidate wrappers are deterministic planner-interface failures, not useful CUDA search
  evidence. Rejecting them before execution protects benchmark time and API budget.

Verification:

- Focused:
  - `uv run pytest tests/test_agent.py::test_parse_variation_decision_rejects_missing_candidate_score_path tests/test_agent.py::test_parse_variation_decision_rejects_missing_candidate_profile_path tests/test_agent.py::test_parse_variation_decision_allows_candidate_path_added_by_patch tests/test_agent.py::test_decision_feedback_explains_missing_candidate_path_error -q`:
    passed, 4 tests.
- Affected:
  - `uv run pytest tests/test_agent.py tests/test_cli.py tests/test_evolve.py -q`: passed, 392
    tests.
- Hygiene:
  - `uv run ruff check`: passed in the runtime repo.
  - `git diff --check`: passed in the runtime repo.

Decision:

- Commit and push the missing-candidate preflight before the next OpenRouter run.

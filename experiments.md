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

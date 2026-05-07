# AVO on Ampere — Architecture Writeup

## Overview

This document describes the architecture of an Agentic Variation Operator (AVO) system targeting attention kernel optimization on the NVIDIA RTX A6000 (Ampere, sm_86). The design follows the AVO formalism introduced by the original paper but adapts the hardware assumptions, baseline targets, and optimization search space to the Ampere microarchitecture. The core claim of AVO — that elevating an LLM agent to serve as the entire variation operator rather than confining it to a single-turn generation step produces qualitatively better optimization outcomes — motivates every structural decision described here.

---

## Hardware Context

The target hardware is the NVIDIA RTX A6000, an Ampere-class GPU (sm_86) with 48GB GDDR6 memory and approximately 310 TFLOPS of BF16 throughput. Critically, Ampere lacks the hardware primitives that define the FA4/Blackwell optimization landscape: there is no Tensor Memory Accelerator (TMA), no WGMMA instruction set, and no 4th-generation tensor core pipeline. The operative hardware primitives are `cp.async` for asynchronous global-to-shared memory copies, `mma.sync` for warp-level matrix multiply-accumulate via 3rd-generation tensor cores, and `__shfl_xor_sync` for warp-level reductions. The register file model and shared memory bank structure are also distinct from Blackwell. This means the Blackwell-specific optimizations analyzed in the original paper — TMA-based load scheduling, WGMMA pipeline overlap, and Blackwell-specific register budgets — do not transfer. However, an equally rich optimization space exists at the Ampere level, and it has not previously been explored by an autonomous agent.

The baseline against which AVO evolves is **FlashAttention-2**, the canonical Ampere-optimized attention implementation. FA2 targets sm_80/sm_86 via `cp.async` pipelining and `mma.sync`, making it the correct seed and comparison point. FlashAttention-4 is architecturally incompatible with sm_86 and is excluded.

---

## System Architecture

The system operates across three nested levels with distinct timescales and responsibilities.

### Level 3 — Continuous Evolution (hours to overnight)

The outermost loop maintains the committed lineage `𝒫ₜ = {(x₀, f(x₀)), (x₁, f(x₁)), ..., (xₜ, f(xₜ))}` as a git repository where each commit encodes one kernel version alongside its structured benchmark score. This level is responsible for:

- Persisting the full evolutionary history in a queryable, diffable form
- Enforcing the commit gate: a candidate `xₜ₊₁` enters the lineage only if it passes correctness and matches or improves the best benchmark score
- Running the stagnation supervisor as a background watchdog
- Maintaining the association between source versions and their multi-dimensional performance vectors

The population structure is **single-lineage** — one chain of committed improvements rather than a branching archive. This is the simplest regime and isolates the effect of the agentic operator itself. Population-level branching (multiple parallel lineages on distinct hardware configurations or distinct optimization axes) is a natural extension but is out of scope here.

### Level 2 — Variation Step (minutes to hours)

A single variation step produces one committed version `xₜ₊₁` from the current lineage. This is not a single LLM call — it is an autonomous multi-session agent run that may internally execute the Level 1 loop dozens of times before committing. The variation step has a natural three-phase structure:

**Reconnaissance.** The agent reads across `𝒫ₜ` — not just the most recent version but comparing profiler characteristics, score trajectories, and implementation strategies across the lineage. It consults `𝒦` to ground its hypotheses in hardware constraints. The agent decides which prior versions are relevant; this decision is not made by the framework.

**Search.** The agent iterates through the Level 1 edit-compile-eval-diagnose loop, generating and testing candidate implementations. Failed candidates remain in the agent's working memory but are not committed to `𝒫ₜ`. The agent adapts its strategy within this phase based on compiler output, profiler feedback, and correctness diagnostics — it does not commit blind retries.

**Commitment.** When the agent produces a candidate that passes the commit gate, the variation step ends and `xₜ₊₁` is persisted. If the agent exhausts its current line of exploration without finding a committable candidate, the step ends without a commit and the supervisor is notified.

### Level 1 — Agent Loop Iteration (seconds to minutes)

The atomic unit of the system. One cycle: read context → form hypothesis → implement edit → compile → correctness check → benchmark → diagnose. The agent owns the control flow entirely — it decides when to exit the loop, when to abandon a direction, and when a candidate is worth committing. The framework provides the tools (compiler, benchmark harness, file system, git) but does not prescribe the order or number of iterations.

---

## The Three Inputs to the Agent

The agent operates over three inputs, formalized as `Agent(𝒫ₜ, 𝒦, f)`.

### Lineage 𝒫ₜ

The committed kernel history. Each entry is a `(source, score_vector)` pair accessible via git. The score vector is the multi-dimensional benchmark result across all evaluated configurations. The agent reads this to understand what has been tried, what the performance trajectory looks like, and which bottlenecks remain unaddressed. The lineage grows monotonically — only improving commits enter it — so it represents a clean record of genuine progress rather than the full internal search tree.

### Knowledge Base 𝒦

A static corpus of Ampere-relevant documentation: the Ampere architecture whitepaper, `cp.async` and `mma.sync` PTX reference, CUDA C++ programming guide sections on shared memory layout and register file behavior, FA2 source code, and relevant CUTLASS Ampere kernel examples. The agent queries this reactively — it reads what it needs when it needs it, rather than having the entire corpus pre-loaded into context. This reactive access pattern means the knowledge base can be large without bloating every LLM call.

### Scoring Function f

The environment's feedback signal. `f` is a vector-valued function:

```
f(xᵢ) = (correctness, tflops_4k_causal, tflops_8k_causal, ..., tflops_32k_noncausal)
```

Correctness is a hard gate — a kernel that produces numerically incorrect output receives zero score on all throughput dimensions regardless of its speed. The reference for correctness is PyTorch's `scaled_dot_product_attention`. The throughput dimensions span sequence lengths `{4096, 8192, 16384, 32768}` under both causal and non-causal masking, with fixed total token count of 32,768 (batch size inversely proportional to sequence length). Each dimension is reported in TFLOPS using standard attention FLOP counts. The vector structure is intentional: it gives the agent the ability to observe that a change degrades short-sequence performance while improving long-sequence performance, and reason about the architectural cause.

---

## Memory Architecture

The agent has two memory systems operating at different scopes.

### External Memory — The Committed Lineage

Git is the persistence layer. Every committed kernel version is a git commit with a structured score payload in the commit metadata. This memory is durable across agent restarts, inspectable by humans, and directly queryable by the agent via shell tools. Because git preserves diffs, the agent can also read *how* each version changed from its predecessor — a richer signal than just the source and score.

### Internal Memory — Conversation History

The agent's working memory is its conversation history: the accumulated sequence of tool calls, compiler outputs, profiler results, correctness failures, and reasoning across the entire run. This is what makes the agent's behavior coherent over long runs — it remembers why it abandoned a direction three hours ago, what error message it saw when it tried a particular register allocation, and what the profiler said about shared memory bank conflicts in version 12.

The conversation history is ephemeral relative to the git lineage — it does not survive a restart. For overnight runs, context window management is a real constraint: a multi-hour agent session accumulates substantial context, and naive accumulation eventually degrades reasoning quality or hits model limits. A compaction strategy — periodic summarization of old turns while preserving full fidelity for recent turns — is necessary for runs beyond a few hours.

---

## Supervisor and Stagnation Recovery

The supervisor is a secondary process that monitors the committed lineage and intervenes when the main agent stalls. Two stagnation signatures trigger intervention:

**Exhaustion.** The agent explores many directions internally across a variation step but produces no commit. This indicates it has mined out its current optimization hypothesis and needs a strategy reset.

**Cycling.** The agent repeatedly attempts the same class of modification — identifiable by reading its conversation history — with no progress. Continued iteration is wasteful.

When triggered, the supervisor reads the full committed lineage trajectory, identifies what classes of optimization have and haven't been attempted, and injects a structured prompt into the agent's context suggesting several new candidate directions. This is not prescriptive — the agent is not told to implement a specific change, only given fresh strategic framing. The supervisor fires conditionally and infrequently; it is a recovery mechanism, not a scheduler.

---

## Ampere Optimization Search Space

The agent is not told which optimizations to pursue. However, the Ampere architecture defines the space of what is achievable, and the knowledge base orients the agent toward it. The principal dimensions of the search space on sm_86:

**`cp.async` pipeline depth and staging.** FA2 uses double-buffered async copies for Q/K/V tiles. Deeper pipelines (triple-buffering) or restructured staging can hide more memory latency, particularly at longer sequence lengths where the K-block loop dominates.

**Shared memory layout and bank conflict elimination.** The placement of Q, K, V, and output tiles in shared memory determines whether 32-bank shared memory accesses are conflict-free. Swizzled layouts trade layout complexity for bank conflict reduction on the `mma.sync` input operands.

**Register pressure management.** The Ampere register file is a fixed per-SM budget partitioned across warps. Kernels that spill accumulators to local memory pay a significant latency penalty. The allocation of registers across the online softmax state (running maximum, running sum) versus the output accumulator `O` is a tunable tradeoff.

**Warp-level softmax reduction patterns.** The online softmax algorithm requires warp-level reductions for the running row-maximum and row-sum. The choice of `__shfl_xor_sync` reduction tree structure and the granularity at which reductions are issued relative to MMA execution affect occupancy and instruction throughput.

**Instruction scheduling around `mma.sync` latency.** `mma.sync` has non-trivial latency on Ampere tensor cores. The interleaving of softmax arithmetic, memory operations, and MMA instructions in the instruction stream determines whether the tensor cores are kept fed or stall waiting for softmax to complete.

**Split-Q vs split-K parallelism.** FA2 uses split-K (each thread block processes a chunk of key/value tokens for a fixed set of query tokens). Split-Q variants (each thread block processes a chunk of query tokens) have different shared memory and register access patterns and may be preferable at certain sequence length and head count configurations.

---

## What This System Is Not

A few explicit non-goals worth stating:

- **Not a hyperparameter tuner.** The agent is not performing grid search over a fixed parameter space (tile sizes, unroll factors). It is making qualitative architectural decisions — whether to restructure the pipeline, how to lay out shared memory, whether to eliminate a branch — that require reasoning about hardware behavior.

- **Not a search over compilation flags.** Compiler-level autotuning (pragma unroll, launch bounds, occupancy hints) is in scope as one class of modification, but it is a small part of the search space. The interesting results come from structural kernel changes.

- **Not a single-GPU benchmark race.** The goal of an overnight run on A6000 is not to claim a world record. It is to demonstrate that the AVO loop produces genuine hardware-level reasoning and discovers non-obvious optimizations autonomously. The metric is whether the committed lineage shows real improvement over FA2 with interpretable causation, not absolute TFLOPS.

---

## Summary of Design Decisions

| Dimension | Choice | Rationale |
|---|---|---|
| Target hardware | A6000 (sm_86, Ampere) | Available hardware |
| Seed kernel | FlashAttention-2 | Only Ampere-compatible SOTA baseline |
| Comparison baseline | FlashAttention-2 | Same reason |
| Population structure | Single lineage | Isolates operator effect |
| Lineage persistence | Git | Durable, diffable, agent-queryable |
| Agent memory | Conversation history | No external retrieval needed at this scale |
| Scoring | Vector-valued, correctness-gated | Preserves per-config signal, prevents fast-but-wrong exploits |
| KB access | Reactive (on-demand reads) | Keeps per-call context lean |
| Supervisor trigger | Elapsed time without commit | Simple, interpretable |
| Context management | Periodic compaction | Required for overnight runs |
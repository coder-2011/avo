

AVO: Agentic Variation Operators for
## Autonomous Evolutionary Search
## Terry Chen
## ∗
## , Zhifan Ye
## ∗
## , Bing Xu
## ∗
## , Zihao Ye, Timmy Liu, Ali Hassani, Tianqi Chen
Andrew Kerr, Haicheng Wu, Yang Xu, Yu-Jung Chen, Hanfeng Chen, Aditya Kane
Ronny Krashinsky, Ming-Yu Liu, Vinod Grover, Luis Ceze, Roger Bringmann,
## John Tran, Wei Liu, Fung Xie, Michael Lightstone, Humphrey Shi
## NVIDIA
## Abstract
Agentic Variation Operators (AVO) are a new family of evolutionary variation
operators that replace the fixed mutation, crossover, and hand-designed heuristics
of classical evolutionary search with autonomous coding agents.   Rather than
confining a language model to candidate generation within a prescribed pipeline,
AVO instantiates variation as a self-directed agent loop that can consult the
current lineage, a domain-specific knowledge base, and execution feedback to
propose, repair, critique, and verify implementation edits.  We evaluate AVO on
attention, among the most aggressively optimized kernel targets in AI, on NVIDIA
Blackwell (B200) GPUs.  Over 7 days of continuous autonomous evolution on
multi-head attention, AVO discovers kernels that outperform cuDNN by up to
3.5% and FlashAttention-4 by up to 10.5% across the evaluated configurations.
The discovered optimizations transfer readily to grouped-query attention, requiring
only 30 minutes of additional autonomous adaptation and yielding gains of up to
7.0% over cuDNN and 9.3% over FlashAttention-4. Together, these results show
that agentic variation operators move beyond prior LLM-in-the-loop evolutionary
pipelines by elevating the agent from candidate generator to variation operator, and
can discover performance-critical micro-architectural optimizations that produce
kernels surpassing state-of-the-art expert-engineered attention implementations on
today’s most advanced GPU hardware.
## 1    Introduction
Large language models have emerged as powerful components in evolutionary search, replacing
hand-crafted mutation operators [1] with learned code generation [2–5]. In these systems, an LLM
generates candidate solutions conditioned on selected parents, while a surrounding framework, which
is usually heuristic-based, handles parent sampling, evaluation, and population management. This
combination has produced notable results in mathematical optimization and algorithm discovery,
including flagship systems such as FunSearch and AlphaEvolve [3,4]. However, confining the LLM
to candidate generation within a prescribed pipeline fundamentally limits what the LLM can discover:
it produces a single output per invocation, with no ability to proactively consult reference materials,
test its changes, interpret feedback, or revise its approach before committing a candidate. For the
most aggressively hand-tuned implementations, where further improvement requires deep, iterative
engineering, this constraint is especially limiting.
We study this problem in the context of attention [6], the central operation in Transformer architectures,
and one of the most heavily optimized GPU kernels. The FlashAttention lineage [7–10] and NVIDIA’s
cuDNN library [11] have pushed attention throughput progressively closer to hardware limits across
## ∗
## Equal Contribution
## Preprint.
arXiv:2603.24517v1  [cs.LG]  25 Mar 2026

## Previous
## Solutions
## Sampling
## LLM
Single-Turn or
## Predefined Workflow
## Evaluation
AI Agent
ToolsMemory
## Previous
## Solutions
## Evaluation
## Utility
## Agent Loop: Planning, Implementing,
## Testing, Debugging
EVO: Classical Evolutionary Search Frameworks
AVO: Agentic Variation Operators
Figure 1: EVO vs AVO: Comparison between prior evolutionary search frameworks (e.g. FunSearch,
AlphaEvolve,  and related LLM-augmented evolutionary approaches) and the proposed Agentic
Variation Operator. Left: Prior approaches follow a fixed pipeline where the LLM is confined to a
single-turn generation step or a predefined workflow, with sampling and evaluation controlled by the
framework. Right: AVO replaces this pipeline with an autonomous AI agent that iteratively plans,
implements, tests, and debugs across long-running sessions, with direct access to previous solutions,
evaluation utilities, tools, and persistent memory.
successive GPU generations, with both FlashAttention-4 (FA4) and cuDNN requiring months of
manual optimization on the latest Blackwell architecture. Surpassing these implementations demands
sustained, iterative interaction with the development environment: studying hardware documentation,
analyzing profiler output to identify bottlenecks, implementing and testing candidate optimizations,
diagnosing correctness failures, and revising strategy based on accumulated experience.
Recent progress in deep agents [12–16] demonstrates that LLMs augmented with planning, persistent
memory, and tool use can autonomously navigate such multi-step engineering workflows, with appli-
cations ranging from resolving complex GitHub issues to generating key deep learning software [17].
This motivates a fundamentally different role for LLMs in evolutionary search: rather than confining
them within a fixed pipeline, we can elevate a deep agent to serve as the variation operator itself. To
this end, we propose Agentic Variation Operators (AVO), in which a self-directed coding agent
replaces the mutation and crossover process in previous works based on single-turn LLMs [3–5] or
fixed workflows [18]. The AVO agent has access to all prior solutions, a domain-specific knowledge
base, and the evaluation utility. It autonomously decides what to consult, what to edit, and when to
evaluate, enabling continuous improvements over extended time horizons.
To demonstrate its effectiveness,  we apply AVO to multi-head attention (MHA) kernels on the
Blackwell B200 GPU, and directly compare against the expert-optimized cuDNN and FlashAttention-
4 kernels. Over 7 days of continuous evolution without human intervention, the agent explored over
500 optimization directions and evolved 40 kernel versions, producing MHA kernels achieving up
to 1668 TFLOPS at BF16 precision, outperforming cuDNN by up to 3.5% and FlashAttention-4
by up to 10.5%.  Our analysis of agent-discovered optimizations reveals that they span multiple
levels of kernel design, including register allocation, instruction pipeline scheduling, and workload
distribution, reflecting genuine hardware-level reasoning. Empirically, we find that the optimization
techniques discovered on MHA transfer effectively to grouped-query attention (GQA): adapting the
evolved MHA kernel to support GQA requires only 30 minutes of additional autonomous agent effort,
yielding up to 7.0% performance improvement over cuDNN and 9.3% over FlashAttention-4.
Our contributions are as follows:
•We introduce Agentic Variation Operators (AVO), a new family of evolutionary variation
operators that elevate the agent from candidate generator to variation operator, autonomously
exploring domain knowledge, implementing edits, and validating results through iterative
interaction with the environment.
•We achieve state-of-the-art MHA throughput on NVIDIA B200 GPUs across the bench-
marked configurations, reaching up to 1668 TFLOPS and outperforming cuDNN by up
to 3.5% and FlashAttention-4 by up to 10.5%. Furthermore, we show that the discovered
optimizations readily transfer to GQA, requiring only 30 minutes of autonomous adaptation
and yielding gains of up to 7.0% over cuDNN and 9.3% over FlashAttention-4.
## 2

•We provide a detailed analysis of the micro-architectural optimizations discovered by the
agent under the benchmarked settings, showing the agent performs genuine hardware-level
reasoning rather than superficial code transformations.
## 2    Background
2.1    Evolutionary Search and Variation Operators
Evolutionary search optimizes over a space of candidates by maintaining a populationPand iteratively
expanding it with new solutions [19]. A population is a set of solution-score pairsP ={(x
i
## ,f(x
i
## ))},
wherefis a scoring function that evaluates each candidate solution. Each iteration produces a new
candidate x
t+1
and updates the population:
## P
t+1
## = Update
## 

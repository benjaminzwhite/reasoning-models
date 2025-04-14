# Reasoning

Experiments with reasoning models, training techniques, papers

### TODO

- build tag system e.g. Policies {DPO, GRPO, ...} with a view to building reading list, integrate with Zotero/Obsidian
- create a tag for "reasoning + interaction with a symbolic system/DSL/program synthesis" type papers

---

## Papers

### Priority

- [**s1: Simple test-time scaling**](https://arxiv.org/abs/2501.19393)
- [**Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling**](https://arxiv.org/abs/2502.06703)
- [**Competitive Programming with Large Reasoning Models**](https://arxiv.org/abs/2502.06807) - OpenAI, "Our findings show that although specialized pipelines such as o1-ioi yield solid improvements, the scaled-up, general-purpose o3 model surpasses those results without relying on hand-crafted inference heuristics. Notably, o3 achieves a gold medal at the 2024 IOI and obtains a Codeforces rating on par with elite human competitors. Overall, these results indicate that scaling general-purpose reinforcement learning, rather than relying on domain-specific techniques, offers a robust path toward state-of-the-art AI in reasoning domains, such as competitive programming."
- [**LLMs Can Easily Learn to Reason from Demonstrations Structure, not content, is what matters!**](https://arxiv.org/abs/2502.07374)
- [**CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction**](https://arxiv.org/abs/2502.07316) - DeepSeek
- [**DAPO: An Open-Source LLM Reinforcement Learning System at Scale**](https://arxiv.org/abs/2503.14476) - ByteDance/Tsinghua
- [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/abs/2504.05118) - ByteDance
- [GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning](https://arxiv.org/abs/2504.02546) - Alibaba, simplifies relative to GRPO, fewer models needed during training also
- [Self-Steering Language Models](https://arxiv.org/abs/2504.07081) - program synthesis related
- [Skywork-OR1 (Open Reasoner 1)](https://github.com/SkyworkAI/Skywork-OR1) - read the nice Notion technical blog via this Github

#### Topic - General or unsorted

- [Technical Report: Enhancing LLM Reasoning with Reward-guided Tree Search](https://arxiv.org/abs/2411.11694)
- [Compressed Chain of Thought: Efficient Reasoning Through Dense Representations](https://arxiv.org/abs/2412.13171)
- [Improving Retrieval Augmented Language Model with Self-Reasoning](https://arxiv.org/abs/2407.19813)
- [Synergy-of-Thoughts: Eliciting Efficient Reasoning in Hybrid Language Models](https://arxiv.org/abs/2402.02563)
- [Disentangling Reasoning Tokens and Boilerplate Tokens For Language Model Fine-tuning](https://arxiv.org/abs/2412.14780)
- [Advancing LLM Reasoning Generalists with Preference Trees](https://arxiv.org/abs/2404.02078v1)
- [Test-time Computing: from System-1 Thinking to System-2 Thinking](https://arxiv.org/abs/2501.02497)
- [Scaling of Search and Learning: A Roadmap to Reproduce o1 from Reinforcement Learning Perspective](https://arxiv.org/abs/2412.14135)
- [BoostStep: Boosting mathematical capability of Large Language Models via improved single-step reasoning](https://arxiv.org/abs/2501.03226)
- [Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Thought](https://arxiv.org/abs/2501.04682)
- [Search-o1: Agentic Search-Enhanced Large Reasoning Models](https://arxiv.org/abs/2501.05366)
- [CoT-based Synthesizer: Enhancing LLM Performance through Answer Synthesis](https://arxiv.org/abs/2501.01668)
- [REL: Working out is all you need](https://arxiv.org/abs/2412.04645)
- [Unconstrained Model Merging for Enhanced LLM Reasoning](https://arxiv.org/abs/2410.13699)
- [MALT: Improving Reasoning with Multi-Agent LLM Training](https://arxiv.org/abs/2412.01928)
- [Reverse Thinking Makes LLMs Stronger Reasoners](https://arxiv.org/abs/2411.19865)
- [Proof of Thought : Neurosymbolic Program Synthesis allows Robust and Interpretable Reasoning](https://arxiv.org/abs/2409.17270)
- [Enhancing Reasoning Capabilities of LLMs via Principled Synthetic Logic Corpus](https://arxiv.org/abs/2411.12498)
- [Evolving Deeper LLM Thinking](https://arxiv.org/abs/2501.09891)
- [Reasoning Language Models: A Blueprint](https://arxiv.org/abs/2501.11223)
- [The Surprising Effectiveness of Test-Time Training for Abstract Reasoning](https://arxiv.org/abs/2411.07279)
- [Skywork-Reward: Bag of Tricks for Reward Modeling in LLMs](https://arxiv.org/abs/2410.18451)
- [Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/abs/2408.15240)
- [Generative Reward Models](https://arxiv.org/abs/2410.12832)
- [Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling](https://arxiv.org/abs/2501.11651)
- [Step-KTO: Optimizing Mathematical Reasoning through Stepwise Binary Feedback](https://arxiv.org/abs/2501.10799)
- [Table as Thought: Exploring Structured Thoughts in LLM Reasoning](https://arxiv.org/abs/2501.02152)
- [B-STaR: Monitoring and Balancing Exploration and Exploitation in Self-Taught Reasoners](https://arxiv.org/abs/2412.17256)
- [Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs](https://arxiv.org/abs/2501.18585)
- [GuardReasoner: Towards Reasoning-based LLM Safeguards](https://arxiv.org/abs/2501.18492)
- [Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search](https://arxiv.org/abs/2502.02508)
- [QLASS: Boosting Language Agent Inference via Q-Guided Stepwise Search](https://arxiv.org/abs/2502.02584)
- [LongDPO: Unlock Better Long-form Generation Abilities for LLMs via Critique-augmented Stepwise Information](https://arxiv.org/abs/2502.02095)
- [LLM+AL: Bridging Large Language Models and Action Languages for Complex Reasoning about Actions](https://arxiv.org/abs/2501.00830)
- [Sample, Scrutinize and Scale: Effective Inference-Time Search by Scaling Verification](https://arxiv.org/abs/2502.01839)
- [ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning](https://arxiv.org/abs/2502.01100)
- [Process Reinforcement through Implicit Rewards](https://arxiv.org/abs/2502.01456) - see also their [HuggingFace blog](https://huggingface.co/blog/ganqu/prime)
- [Towards Learning to Reason: Comparing LLMs with Neuro-Symbolic on Arithmetic Relations in Abstract Reasoning](https://arxiv.org/abs/2412.05586)
- [BOLT: Bootstrap Long Chain-of-Thought in Language Models without Distillation](https://arxiv.org/abs/2502.03860)
- [TypedThinker: Typed Thinking Improves Large Language Model Reasoning](https://arxiv.org/abs/2410.01952)
- [ReasonFlux: Hierarchical LLM Reasoning via Scaling Thought Templates](https://arxiv.org/abs/2502.06772)
- [Exploring the Limit of Outcome Reward for Learning Mathematical Reasoning](https://arxiv.org/abs/2502.06781)
- [CoT-Valve: Length-Compressible Chain-of-Thought Tuning](https://arxiv.org/abs/2502.09601)
- [LLM Pretraining with Continuous Concepts](https://arxiv.org/abs/2502.08524) - follow-up to Coconut paper (Training Large Language Model to Reason in a Continuous Latent Space, see below)
- [LIMO: Less is More for Reasoning](https://arxiv.org/abs/2502.03387)
- [Diverse Inference and Verification for Advanced Reasoning](https://arxiv.org/abs/2502.09955)
- [CRANE: Reasoning with constrained LLM generation](https://arxiv.org/abs/2502.09061)
- [AdaptiveStep: Automatically Dividing Reasoning Step through Model Confidence](https://arxiv.org/abs/2502.13943)
- [OctoTools: An Agentic Framework with Extensible Tools for Complex Reasoning](https://arxiv.org/abs/2502.11271)
- [Thinking Preference Optimization](https://arxiv.org/abs/2502.13173)
- [S2R: Teaching LLMs to Self-verify and Self-correct via Reinforcement Learning](https://arxiv.org/abs/2502.12853)
- [Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning](https://arxiv.org/abs/2502.14768)
- [Reasoning with Reinforced Functional Token Tuning](https://arxiv.org/abs/2502.13389)
- [SIFT: Grounding LLM Reasoning in Contexts via Stickers](https://arxiv.org/abs/2502.14922)
- [How to Get Your LLM to Generate Challenging Problems for Evaluation](https://arxiv.org/abs/2502.14678) - maybe useful for making reasoning datasets
- [Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs](https://arxiv.org/abs/2503.01307)
- [General Reasoning Requires Learning to Reason from the Get-go](https://arxiv.org/abs/2502.19402)
- [CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation](https://arxiv.org/abs/2502.21074)
- [START: Self-taught Reasoner with Tools](https://arxiv.org/abs/2503.04625)
- [Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching](https://arxiv.org/abs/2503.05179)
- [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592)
- [An Empirical Study on Eliciting and Improving R1-like Reasoning Models](https://arxiv.org/abs/2503.04548)
- [Optimizing Test-Time Compute via Meta Reinforcement Fine-Tuning](https://arxiv.org/abs/2503.07572)
- [ϕ-Decoding: Adaptive Foresight Sampling for Balanced Inference-Time Exploration and Exploitation](https://arxiv.org/abs/2503.13288)
- [Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn't](https://arxiv.org/abs/2503.16219)
- [Deconstructing Long Chain-of-Thought: A Structured Reasoning Optimization Framework for Long CoT Distillation](https://arxiv.org/abs/2503.16385)
- [Code to Think, Think to Code: A Survey on Code-Enhanced Reasoning and Reasoning-Driven Code Intelligence in LLMs](https://arxiv.org/abs/2502.19411) -TODO: make a "code enhances reasoning" tag
- [CodeSteer: Symbolic-Augmented Language Models via Code/Text Guidance](https://arxiv.org/abs/2502.04350)
- [I Have Covered All the Bases Here: Interpreting Reasoning Features in Large Language Models via Sparse Autoencoders](https://arxiv.org/abs/2503.18878)
- [SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild](https://arxiv.org/abs/2503.18892)
- [A Probabilistic Inference Approach to Inference-Time Scaling of LLMs using Particle-Based Monte Carlo Methods](https://arxiv.org/abs/2502.01618)
- [Planning In Natural Language Improves LLM Search For Code Generation](https://arxiv.org/abs/2409.03733)
- [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470)
- [MCTS-RAG: Enhancing Retrieval-Augmented Generation with Monte Carlo Tree Search](https://arxiv.org/abs/2503.20757)
- [Unlocking Efficient Long-to-Short LLM Reasoning with Model Merging](https://arxiv.org/abs/2503.20641)
- [CPPO: Accelerating the Training of Group Relative Policy Optimization-Based Reasoning Models](https://arxiv.org/abs/2503.22342)
- [Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning](https://arxiv.org/abs/2502.03275)
- [Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model](https://arxiv.org/abs/2503.24290)
- [Crossing the Reward Bridge: Expanding RL with Verifiable Rewards Across Diverse Domains](https://arxiv.org/abs/2503.23829)
- [Z1: Efficient Test-time Scaling with Code](https://arxiv.org/abs/2504.00810)
- [Recitation over Reasoning: How Cutting-Edge Language Models Can Fail on Elementary School-Level Reasoning Problems?](https://arxiv.org/abs/2504.00509) - critical of reasoning results, similar to SimpleBench
- [Reasoning-SQL: Reinforcement Learning with SQL Tailored Partial Rewards for Reasoning-Enhanced Text-to-SQL](https://arxiv.org/abs/2503.23157)
- [JudgeLRM: Large Reasoning Models as a Judge](https://arxiv.org/abs/2504.00050)
- [When More is Less: Understanding Chain-of-Thought Length in LLMs](https://arxiv.org/abs/2502.07266)
- [OpenCodeReasoning: Advancing Data Distillation for Competitive Coding](https://arxiv.org/abs/2504.01943) - mainly a dataset but interesting paper also
- [Bag of Tricks for Inference-time Computation of LLM Reasoning](https://arxiv.org/abs/2502.07191) - looks nice, practical stuff
- [GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning](https://arxiv.org/abs/2504.00891)
- [Sample, Don't Search: Rethinking Test-Time Alignment for Language Models](https://arxiv.org/abs/2504.03790)
- [Rethinking Reflection in Pre-Training](https://arxiv.org/abs/2504.04022)
- [Generative Evaluation of Complex Reasoning in Large Language Models](https://arxiv.org/abs/2504.02810) - synthetic benchmark/evaluation
- [MastermindEval: A Simple But Scalable Reasoning Benchmark](https://arxiv.org/abs/2503.05891) - benchmark
- [A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibility](https://arxiv.org/abs/2504.07086) - critical of some benchmarks, evals and methodology
- [Large Language Models Meet Symbolic Provers for Logical Reasoning Evaluation](https://arxiv.org/abs/2502.06563) - ICLR 2025
- [Seed-Thinking-v1.5: Advancing Superb Reasoning Models with Reinforcement Learning](https://github.com/ByteDance-Seed/Seed-Thinking-v1.5) - ByteDance, paper is available via this Github
- [TextGames: Learning to Self-Play Text-Based Puzzle Games via Language Model Reasoning](https://arxiv.org/abs/2502.18431)
- [Complex LLM Planning via Automated Heuristics Discovery](https://arxiv.org/abs/2502.19295) - code enhanced reasoning related, generates heuristics as Python functions

#### Topic - Surveys and reviews

- [Towards Large Reasoning Models: A Survey of Reinforced Reasoning with Large Language Models](https://arxiv.org/abs/2501.09686)
- [Reinforcement Learning Enhanced LLMs: A Survey](https://arxiv.org/abs/2412.10400)
- [From System 1 to System 2: A Survey of Reasoning Large Language Models](https://arxiv.org/abs/2502.17419)
- [What, How, Where, and How Well? A Survey on Test-Time Scaling in Large Language Models](https://arxiv.org/abs/2503.24235) - section 4.1 is on reasoning specifically but general paper is nicely organized and relevant

#### Topic - Reasoning + Agents

- [ATLaS: Agent Tuning via Learning Critical Steps](https://arxiv.org/abs/2503.02197)
- [Agent models: Internalizing Chain-of-Action Generation into Reasoning models](https://arxiv.org/abs/2503.06580)
- [The Danger of Overthinking: Examining the Reasoning-Action Dilemma in Agentic Tasks](https://arxiv.org/abs/2502.08235)

#### Topic - Information retrieval and search

- [Learning More Effective Representations for Dense Retrieval through Deliberate Thinking Before Search](https://arxiv.org/abs/2502.12974)
- [O1 Embedder: Let Retrievers Think Before Action](https://arxiv.org/abs/2502.07555)
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516)
- [Open Deep Search: Democratizing Search with Open-source Reasoning Agents](https://arxiv.org/abs/2503.20201)
- [Distillation and Refinement of Reasoning in Small Language Models for Document Re-ranking](https://arxiv.org/abs/2504.03947) - uses BRIGHT benchmark
- [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160) - deep research, does have a reasoning type model involved for planning stage

#### Topic - X of Thoughts variants

- [Path-of-Thoughts: Extracting and Following Paths for Robust Relational Reasoning with Large Language Models](https://arxiv.org/abs/2412.17963)
- [Forest-of-Thought: Scaling Test-Time Compute for Enhancing LLM Reasoning](https://arxiv.org/abs/2412.09078)

#### Topic - Efficiency, Decoding Strategies, Implementation Tricks etc.

- [Efficiently Serving LLM Reasoning Programs with Certaindex](https://arxiv.org/abs/2412.20993)
- [Path-Consistency: Prefix Enhancement for Efficient Inference in LLM](https://arxiv.org/abs/2409.01281)
- [Reward-Guided Speculative Decoding for Efficient LLM Reasoning](https://arxiv.org/abs/2501.19324)
- [LightThinker: Thinking Step-by-Step Compression](https://arxiv.org/abs/2502.15589)
- [Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models](https://arxiv.org/abs/2503.16419)
- [A Survey of Efficient Reasoning for Large Reasoning Models: Language, Multimodality, and Beyond](https://arxiv.org/abs/2503.21614) - their Github page is nice also: [https://github.com/XiaoYee/Awesome_Efficient_LRM_Reasoning](https://github.com/XiaoYee/Awesome_Efficient_LRM_Reasoning)
- [Harnessing the Reasoning Economy: A Survey of Efficient Reasoning for Large Language Models](https://arxiv.org/abs/2503.24377) - Github [https://github.com/DevoAllen/Awesome-Reasoning-Economy-Papers](https://github.com/DevoAllen/Awesome-Reasoning-Economy-Papers)
- [Efficient Inference for Large Reasoning Models: A Survey](https://arxiv.org/abs/2503.23077) - Github [https://github.com/yueliu1999/Awesome-Efficient-Inference-for-LRMs](https://github.com/yueliu1999/Awesome-Efficient-Inference-for-LRMs)
- [Hawkeye:Efficient Reasoning with Model Collaboration](https://arxiv.org/abs/2504.00424) - uses a Qwen 0.5B SLM to speed up reasoning

#### Topic - Ensembling, Boosting, Stacking etc.

- [Ensembling Large Language Models with Process Reward-Guided Tree Search for Better Complex Reasoning](https://arxiv.org/abs/2412.15797)

#### Topic - Evaluation of reasoning

- [Are Your LLMs Capable of Stable Reasoning?](https://arxiv.org/abs/2412.13147)
- [The Jumping Reasoning Curve? Tracking the Evolution of Reasoning Performance in GPT-\[n\] and o-\[n\] Models on Multimodal Puzzles](https://arxiv.org/abs/2502.01081)

#### Topic - Mathematics

- [rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking](https://arxiv.org/abs/2501.04519)
- [The Lessons of Developing Process Reward Models in Mathematical Reasoning](https://arxiv.org/abs/2501.07301)
- [AceMath: Advancing Frontier Math Reasoning with Post-Training and Reward Modeling](https://arxiv.org/abs/2412.15084)
- [LLaMA-Berry: Pairwise Optimization for O1-like Olympiad-Level Mathematical Reasoning](https://arxiv.org/abs/2410.02884)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) - see also the [Yannic Kilcher paper review](https://www.youtube.com/watch?v=bAWV_yrqx4w)
- [MetaLadder: Ascending Mathematical Solution Quality via Analogical-Problem Reasoning Transfer](https://arxiv.org/abs/2503.14891)
- [Leanabell-Prover: Posttraining Scaling in Formal Reasoning](https://arxiv.org/abs/2504.06122)
- [FANS -- Formal Answer Selection for Natural Language Math Reasoning Using Lean4](https://arxiv.org/abs/2503.03238)
- [LeanProgress: Guiding Search for Neural Theorem Proving via Proof Progress Prediction](https://arxiv.org/abs/2502.17925)

#### Topic - Coding

- [ACECODER: Acing Coder RL via Automated Test-Case Synthesis](https://arxiv.org/abs/2502.01718)
- [S*: Test Time Scaling for Code Generation](https://arxiv.org/abs/2502.14382)
- [Learning to Solve and Verify: A Self-Play Framework for Code and Test Generation](https://arxiv.org/abs/2502.14948) - not directly about reasoning but nice approach
- [Think Like Human Developers: Harnessing Community Knowledge for Structured Code Reasoning](https://arxiv.org/abs/2503.14838)

#### Topic - Other specialized reasoning areas (medical, legal, etc.)

- [ChemAgent: Self-updating Library in Large Language Models Improves Chemical Reasoning](https://arxiv.org/abs/2501.06590)
- [MedS3: Towards Medical Small Language Models with Self-Evolved Slow Thinking](https://arxiv.org/abs/2501.12051)
- [MedAgentsBench: Benchmarking Thinking Models and Agent Frameworks for Complex Medical Reasoning](https://arxiv.org/abs/2503.07459)
- [Fin-R1: A Large Language Model for Financial Reasoning through Reinforcement Learning](https://arxiv.org/abs/2503.16252)
- [m1: Unleash the Potential of Test-Time Scaling for Medical Reasoning with Large Language Models](https://arxiv.org/abs/2504.00869)

#### Topic - Multimodal

TODO: expand with separate tags

- [Multimodal Chain-of-Thought Reasoning: A Comprehensive Survey](https://arxiv.org/abs/2503.12605)
- [https://embodied-reasoner.github.io/](https://embodied-reasoner.github.io/)
- [Skywork R1V: Pioneering Multimodal Reasoning with Chain-of-Thought](https://arxiv.org/abs/2504.05599)

#### Topic - Adjacent subjects, maybe related

Personal reading notes/papers that gave me some reasoning-related ideas

- [Answer Set Networks: Casting Answer Set Programming into Deep Learning](https://arxiv.org/abs/2412.14814)
- [UQE: A Query Engine for Unstructured Databases](https://arxiv.org/abs/2407.09522)
- [LTNtorch: PyTorch Implementation of Logic Tensor Networks](https://arxiv.org/abs/2409.16045)
- [SymBa: Symbolic Backward Chaining for Structured Natural Language Reasoning](https://arxiv.org/abs/2402.12806)
- [Shifting Long-Context LLMs Research from Input to Output](https://arxiv.org/abs/2503.04723) - generating large reasoning chains
- [BlendRL: A Framework for Merging Symbolic and Neural Policy Learning](https://arxiv.org/abs/2410.11689) - ICLR 2025
- [Critical Thinking: Which Kinds of Complexity Govern Optimal Reasoning Length?](https://arxiv.org/abs/2504.01935) - more theoretical but maybe interesting

---

## Datasets

- [https://huggingface.co/datasets/open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) - OpenR1-Math-220k is a large-scale dataset for mathematical reasoning. It consists of 220k math problems with two to four reasoning traces generated by DeepSeek R1 for problems from NuminaMath 1.5. The traces were verified using Math Verify for most samples and Llama-3.3-70B-Instruct as a judge for 12% of the samples, and each problem contains at least one reasoning trace with a correct answer.
- [https://huggingface.co/collections/open-r1/reasoning-datasets-67980cac6e816a0eda98c678](https://huggingface.co/collections/open-r1/reasoning-datasets-67980cac6e816a0eda98c678) - Datasets with reasoning traces for math and code released by the community

---

## Benchmarks

- [The CLRS-Text Algorithmic Reasoning Language Benchmark](https://arxiv.org/abs/2406.04229v1)
- [LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks](https://arxiv.org/abs/2412.15204)
- [BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval](https://arxiv.org/abs/2407.12883)
- [NPHardEval: Dynamic Benchmark on Reasoning Ability of Large Language Models via Complexity Classes](https://arxiv.org/abs/2312.14890)
- [LogicBench: Towards Systematic Evaluation of Logical Reasoning Ability of Large Language Models](https://arxiv.org/abs/2404.15522v2)
- [PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models](https://arxiv.org/abs/2501.03124)
- [ProcessBench: Identifying Process Errors in Mathematical Reasoning](https://arxiv.org/abs/2412.06559)
- [ToolComp: A Multi-Tool Reasoning & Process Supervision Benchmark](https://arxiv.org/abs/2501.01290)
- [PhD Knowledge Not Required: A Reasoning Challenge for Large Language Models](https://arxiv.org/abs/2502.01584)
- [HARP: A challenging human-annotated math reasoning benchmark](https://arxiv.org/abs/2412.08819)
- [SPIN-Bench: How Well Do LLMs Plan Strategically and Reason Socially?](https://arxiv.org/abs/2503.12349)
- [CURIE: Evaluating LLMs On Multitask Scientific Long Context Understanding and Reasoning](https://arxiv.org/abs/2503.13517) - Google, ICLR 2025
- [QuestBench: Can LLMs ask the right question to acquire information in reasoning tasks?](https://arxiv.org/abs/2503.22674)
- [CodeARC: Benchmarking Reasoning Capabilities of LLM Agents for Inductive Program Synthesis](https://arxiv.org/abs/2503.23145)
- [CrossWordBench: Evaluating the Reasoning Capabilities of LLMs and LVLMs with Controllable Puzzle Generation](https://arxiv.org/abs/2504.00043)
- [FINEREASON: Evaluating and Improving LLMs' Deliberate Reasoning through Reflective Puzzle Solving](https://arxiv.org/abs/2502.20238)
- [LR2Bench: Evaluating Long-chain Reflective Reasoning Capabilities of Large Language Models via Constraint Satisfaction Problems](https://arxiv.org/abs/2502.17848)

---

## Blogs and articles

- [HKUST - Simple Reinforcement Learning for Reasoning](https://hkust-nlp.notion.site/simplerl-reason)
- [OpenAI - Deliberative alignment: reasoning enables safer language models](https://openai.com/index/deliberative-alignment/)
- [DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) - main project page is [https://agentica-project.com/](https://agentica-project.com/), HuggingFace page is [https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview)
- [Open R1: Update #3](https://huggingface.co/blog/open-r1/update-3) - HuggingFace Open-R1 update on Codeforces and IOI 24 Informatics Olympiads, datasets etc.
- [Understanding R1-Zero-Like Training: A Critical Perspective](https://github.com/sail-sg/understand-r1-zero)
- [YouTube - Nathan Lambert - GRPO's new variants and implementation secrets](https://www.youtube.com/watch?v=amrJDwMUFNs)
- [Reasoning models don't always say what they think](https://www.anthropic.com/research/reasoning-models-dont-say-think) - nice Anthropic blog, basically they take a reasoning model and place hints to correct answer in the input prompts (during eval), then measure whether model mentions it used the hint to reach its final answer etc.

#### Workshops

- [Reasoning and Planning for Large Language Models, ICLR 2025, April 28 2025, Singapore](https://workshop-llm-reasoning-planning.github.io/) - 

---

## Repos

- [**HuggingFace Open R1 - A fully open reproduction of DeepSeek-R1**](https://github.com/huggingface/open-r1)
- [LLaMA-O1: Open Large Reasoning Model Frameworks For Training, Inference and Evaluation With PyTorch and HuggingFace](https://github.com/SimpleBerry/LLaMA-O1/)
- [Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions](https://github.com/AIDC-AI/Marco-o1)
- [veRL: Volcano Engine Reinforcement Learning for LLM](https://github.com/volcengine/verl)
- [Recipes to scale inference-time compute of open models](https://github.com/huggingface/search-and-learn)
- [Facebook Research COCONUT paper - Training Large Language Model to Reason in a Continuous Latent Space](https://github.com/facebookresearch/coconut)
- [Simple Reinforcement Learning for Reasoning - replication DeepSeek-R1-Zero and DeepSeek-R1 training on small models with limited data](https://github.com/hkust-nlp/simpleRL-reason)
- [The Path to Open-Sourcing the DeepSeek Inference Engine](https://github.com/deepseek-ai/open-infra-index/tree/main/OpenSourcing_DeepSeek_Inference_Engine) - DeepSeek Open Infra Github, as of April 2025 they announce that they will open source their inference engine

#### Collections

- [Linked from paper https://arxiv.org/abs/2501.02497 - This repository contains the resources for Test-time Computing: from System-1 Thinking to System-2 Thinking](https://github.com/Dereck0602/Awesome_Test_Time_LLMs)
- [Curated collection of papers and resources on how to unlock the reasoning ability of LLMs and MLLMs.](https://github.com/atfortes/Awesome-LLM-Reasoning)
- [A curated list of language modeling researches for code (and other software engineering activities), plus related datasets.](https://github.com/codefuse-ai/Awesome-Code-LLM) - has related subjects around "code enhances reasoning" general area

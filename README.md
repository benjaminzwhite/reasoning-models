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

#### Topic - Surveys and reviews

- [Towards Large Reasoning Models: A Survey of Reinforced Reasoning with Large Language Models](https://arxiv.org/abs/2501.09686)
- [Reinforcement Learning Enhanced LLMs: A Survey](https://arxiv.org/abs/2412.10400)
- [From System 1 to System 2: A Survey of Reasoning Large Language Models](https://arxiv.org/abs/2502.17419)

#### Topic - Information retrieval

- [Learning More Effective Representations for Dense Retrieval through Deliberate Thinking Before Search](https://arxiv.org/abs/2502.12974)
- [O1 Embedder: Let Retrievers Think Before Action](https://arxiv.org/abs/2502.07555)

#### Topic - X of Thoughts variants

- [Path-of-Thoughts: Extracting and Following Paths for Robust Relational Reasoning with Large Language Models](https://arxiv.org/abs/2412.17963)
- [Forest-of-Thought: Scaling Test-Time Compute for Enhancing LLM Reasoning](https://arxiv.org/abs/2412.09078)

#### Topic - Efficiency, Decoding Strategies, Implementation Tricks etc.

- [Efficiently Serving LLM Reasoning Programs with Certaindex](https://arxiv.org/abs/2412.20993)
- [Path-Consistency: Prefix Enhancement for Efficient Inference in LLM](https://arxiv.org/abs/2409.01281)
- [Reward-Guided Speculative Decoding for Efficient LLM Reasoning](https://arxiv.org/abs/2501.19324)
- [LightThinker: Thinking Step-by-Step Compression](https://arxiv.org/abs/2502.15589)

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

#### Topic - Coding

- [ACECODER: Acing Coder RL via Automated Test-Case Synthesis](https://arxiv.org/abs/2502.01718)
- [S*: Test Time Scaling for Code Generation](https://arxiv.org/abs/2502.14382)

#### Topic - Other specialized reasoning areas (medical, legal, etc.)

- [ChemAgent: Self-updating Library in Large Language Models Improves Chemical Reasoning](https://arxiv.org/abs/2501.06590)
- [MedS3: Towards Medical Small Language Models with Self-Evolved Slow Thinking](https://arxiv.org/abs/2501.12051)

#### Topic - Adjacent subjects, maybe related

- [Answer Set Networks: Casting Answer Set Programming into Deep Learning](https://arxiv.org/abs/2412.14814)
- [UQE: A Query Engine for Unstructured Databases](https://arxiv.org/abs/2407.09522)
- [LTNtorch: PyTorch Implementation of Logic Tensor Networks](https://arxiv.org/abs/2409.16045)
- [SymBa: Symbolic Backward Chaining for Structured Natural Language Reasoning](https://arxiv.org/abs/2402.12806)
- [ATLaS: Agent Tuning via Learning Critical Steps](https://arxiv.org/abs/2503.02197)


## Datasets

- [https://huggingface.co/datasets/open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) - OpenR1-Math-220k is a large-scale dataset for mathematical reasoning. It consists of 220k math problems with two to four reasoning traces generated by DeepSeek R1 for problems from NuminaMath 1.5. The traces were verified using Math Verify for most samples and Llama-3.3-70B-Instruct as a judge for 12% of the samples, and each problem contains at least one reasoning trace with a correct answer.

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

## Blogs and articles

- [HKUST - Simple Reinforcement Learning for Reasoning](https://hkust-nlp.notion.site/simplerl-reason)
- [OpenAI - Deliberative alignment: reasoning enables safer language models](https://openai.com/index/deliberative-alignment/)
- [DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) - main project page is [https://agentica-project.com/](https://agentica-project.com/), HuggingFace page is [https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview)

## Repos

- [**HuggingFace Open R1 - A fully open reproduction of DeepSeek-R1**](https://github.com/huggingface/open-r1)
- [LLaMA-O1: Open Large Reasoning Model Frameworks For Training, Inference and Evaluation With PyTorch and HuggingFace](https://github.com/SimpleBerry/LLaMA-O1/)
- [Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions](https://github.com/AIDC-AI/Marco-o1)
- [veRL: Volcano Engine Reinforcement Learning for LLM](https://github.com/volcengine/verl)
- [Recipes to scale inference-time compute of open models](https://github.com/huggingface/search-and-learn)
- [Facebook Research COCONUT paper - Training Large Language Model to Reason in a Continuous Latent Space](https://github.com/facebookresearch/coconut)
- [Simple Reinforcement Learning for Reasoning - replication DeepSeek-R1-Zero and DeepSeek-R1 training on small models with limited data](https://github.com/hkust-nlp/simpleRL-reason)

#### Collections

- [Linked from paper https://arxiv.org/abs/2501.02497 - This repository contains the resources for Test-time Computing: from System-1 Thinking to System-2 Thinking](https://github.com/Dereck0602/Awesome_Test_Time_LLMs)
- [Curated collection of papers and resources on how to unlock the reasoning ability of LLMs and MLLMs.](https://github.com/atfortes/Awesome-LLM-Reasoning)

# Reasoning

Experiments with reasoning models, training techniques, papers

### TODO

- build tag system e.g. Policies {DPO, GRPO, ...} with a view to building reading list, integrate with Zotero/Obsidian
- create a tag for "reasoning + interaction with a symbolic system/DSL/program synthesis" type papers
- use new approach for formatting table and generating summary: test time and cost first

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
- [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536) - ByteDance, seems to be like a Code Action training but need to read in detail
- [ToRL: Scaling Tool-Integrated RL](https://arxiv.org/abs/2503.23383) - TODO: create a Tool Use RL general topic
- [**Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?**](https://arxiv.org/abs/2504.13837) - Tsinghua/SJTU, critical analysis of RLVR reinforcement learning with verifiable rewards
- [**Generative AI Act II: Test Time Scaling Drives Cognition Engineering**](https://arxiv.org/abs/2504.13828) - 76 pages, really nice overview of all the related topics, has code and mini tutorials also
- [TTRL: Test-Time Reinforcement Learning](https://arxiv.org/abs/2504.16084)
- [Enhancing LLM Reasoning with Iterative DPO: A Comprehensive Empirical Investigation](https://arxiv.org/abs/2503.12854) - read this with the Tsinghua/SJTU paper above; seems using a strong base model and some DPO works to boost reasoning performance
- [Reinforcement Learning for Reasoning in Large Language Models with One Training Example](https://arxiv.org/abs/2504.20571)
- [WebThinker: Empowering Large Reasoning Models with Deep Research Capability](https://arxiv.org/abs/2504.21776) - nice Deep Research implementation, has Github [https://github.com/RUC-NLPIR/WebThinker](https://github.com/RUC-NLPIR/WebThinker)
- [100 Days After DeepSeek-R1: A Survey on Replication Studies and More Directions for Reasoning Language Models](https://arxiv.org/abs/2505.00551) - nice review
- [RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2504.20073) - reasoning + agents, has Github and code etc.
- [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2505.03335) - has a cool project page also [https://andrewzh112.github.io/absolute-zero-reasoner/](https://andrewzh112.github.io/absolute-zero-reasoner/)
- [X-Reasoner: Towards Generalizable Reasoning Across Modalities and Domains](https://arxiv.org/abs/2505.03981) - Microsoft Research, they train entirely on text and find that reasoning transfers to multimodal (image) performance
- [Putting the Value Back in RL: Better Test-Time Scaling by Unifying LLM Reasoners With Verifiers](https://arxiv.org/abs/2505.04842) - jointly train reasoner and verifier
- [Reinforcement Learning: A Comprehensive Overview](https://arxiv.org/abs/2412.05265) - entire book by Kevin P Murphy (probabilistic machine learning author); not entirely about reasoning models but has nice sections on RL+LLMs and related subjects
- [Seek in the Dark: Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space](https://arxiv.org/abs/2505.13308) - seems to be training-free, need to re-read to be clear
- [lmgame-Bench: How Good are LLMs at Playing Games?](https://arxiv.org/abs/2505.15146) - not entirely about reasoning, but all the best performing models are reasoning models (o3 etc.) and I really like the paper, especially the correlation studies with tasks in Table/Figure 3, and the general area
- [DisCO: Reinforcing Large Reasoning Models with Discriminative Constrained Optimization](https://arxiv.org/abs/2505.12366) - looks like interesting approach/alternative to GRPO, need to implement and test
- [Enigmata: Scaling Logical Reasoning in Large Language Models with Synthetic Verifiable Puzzles](https://arxiv.org/abs/2505.19914) - cool, training with logic puzzles, shows generalization to coding and mathematics; has a benchmark also
- [DeepTheorem: Advancing LLM Reasoning for Theorem Proving Through Natural Language and Reinforcement Learning](https://arxiv.org/abs/2505.23754) - Tencent, nice approach and methods
- [Universal Reasoner: A Single, Composable Plug-and-Play Reasoner for Frozen LLMs](https://arxiv.org/abs/2505.19075) - looks good, approach works with frozen models but I didn't understand training on first reading so need to restudy paper
- [ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models](https://arxiv.org/abs/2505.24864) - Nvidia/Nemotron team, model is here [https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B)
- [REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards](https://arxiv.org/abs/2505.24760) - nice Python library to check out, Github is here [https://github.com/open-thought/reasoning-gym](https://github.com/open-thought/reasoning-gym)
- [Spurious Rewards: Rethinking Training Signals in RLVR](https://rethink-rlvr.notion.site/Spurious-Rewards-Rethinking-Training-Signals-in-RLVR-1f4df34dac1880948858f95aeb88872f) - blog for their paper; "We show that you can do RLVR on Qwen2.5-Math models with completely random or incorrect rewards, and still get massive math benchmark gains."
- [OpenThoughts: Data Recipes for Reasoning Models](https://arxiv.org/abs/2506.04178) - cool project and organization, they have a project page [https://www.openthoughts.ai/](https://www.openthoughts.ai/)
- [Play to Generalize: Learning to Reason Through Game Play](https://arxiv.org/abs/2506.08011) - nice, they find that just by training on video games you get out of domain performance improvements (i.e. on mathematics); it would be cool to see more of the results about "game feature vs. out of domain improvement" (at the end of the paper - basically it seems that if you train on games that have a "spatial" component, the mathematics improves on "geometry" tasks for example)
- [Reinforcement Pre-Training](https://arxiv.org/abs/2506.08007) - from the abstract: "Specifically, we reframe next-token prediction as a reasoning task trained using RL, where it receives verifiable rewards for correctly predicting the next token for a given context. RPT offers a scalable method to leverage vast amounts of text data for general-purpose RL, rather than relying on domain-specific annotated answers."
- [CoRT: Code-integrated Reasoning within Thinking](https://arxiv.org/abs/2506.09820) - USTC/Qwen team
- [Resa: Transparent Reasoning Models via SAEs](https://arxiv.org/abs/2506.09967)
- [ComfyUI-R1: Exploring Reasoning Models for Workflow Generation](https://arxiv.org/abs/2506.09790) - Alibaba; really cool
- [A taxonomy for next-generation reasoning models - Where we've been and where we're going with RLVR](https://www.interconnects.ai/p/next-gen-reasoners) - Nathan Lambert blog; really good overview as of June 2025
- [Scaling Speculative Decoding with Lookahead Reasoning](https://arxiv.org/abs/2506.19830) - Hao lab UCSD; technical, need to re-read details carefully, code is available. " Our key insight is that reasoning models generate step-by-step, and each step needs only to be semantically correct, not exact token matching. In Lookahead Reasoning, a lightweight draft model proposes several future steps; the target model expands each proposal in one batched pass, and a verifier keeps semantically correct steps while letting the target regenerate any that fail. Token-level SD still operates within each reasoning step, so the two layers *(i.e. standard speculative decoding)* of parallelism multiply."
- [Thought Anchors: Which LLM Reasoning Steps Matter?](https://arxiv.org/abs/2506.19143) - Interesting "interpretability" paper; Neel Nanda coauthor. Links to a nice interactive tool also [https://www.thought-anchors.com/](https://www.thought-anchors.com/)
- [WebSailor: Navigating Super-human Reasoning for Web Agent](https://arxiv.org/abs/2507.02592) - awesome, Tongyi Lab Alibaba; "Proprietary agentic systems like DeepResearch have demonstrated superhuman capabilities on extremely complex information-seeking benchmarks such as BrowseComp, a feat previously unattainable. We posit that their success hinges on a sophisticated reasoning pattern absent in open-source models: the ability to systematically reduce extreme uncertainty when navigating vast information landscapes. Based on this insight, we introduce WebSailor, a complete post-training methodology designed to instill this crucial capability. Our approach involves generating novel, high-uncertainty tasks through structured sampling and information obfuscation, RFT cold start, and an efficient agentic RL training algorithm, Duplicating Sampling Policy Optimization (DUPO)."
- [Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities](https://arxiv.org/abs/2507.06261) - Gemini 2.5 technical report
- [Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination](https://arxiv.org/abs/2507.10532) - nice study attemping to clear up some of the confusing results from April-June where people were training models on incorrect or low quality reasoning datasets etc.
- [Towards Agentic RAG with Deep Reasoning: A Survey of RAG-Reasoning Systems in LLMs](https://arxiv.org/abs/2507.09477) - great review
- [Scaling Up RL: Unlocking Diverse Reasoning in LLMs via Prolonged Training](https://arxiv.org/abs/2507.12507) - Nvidia, nice technical report on Nemotron-Research-Reasoning-Qwen-1.5B, they talk about improvements to GRPO (basically seems they validate using DAPO and some implementation changes etc.)
- [**Group Sequence Policy Optimization**](https://arxiv.org/abs/2507.18071) - Qwen team, describes Qwen3 training also
- [Test-Time Scaling with Reflective Generative Model](https://arxiv.org/abs/2507.01951) - looks good but I didn't understand all the implementation details yet; "MetaStone‑S1 is trained based on our proposed reflective generative form, which combines “Long-CoT Reinforcement Learning” and “Process Reward Learning” into a unified training form. This form enables a single model to simultaneously achieve deep reasoning and high-quality reasoning trajectory selection. By sharing the backbone network between the PRMs and policy models, MetaStone‑S1 significantly reduces the inference cost of PRMs by 99%, resulting in faster and higher-quality responses."; Github is [https://github.com/MetaStone-AI/MetaStone-S1](https://github.com/MetaStone-AI/MetaStone-S1)
- [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457)
- [Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty](https://arxiv.org/abs/2507.16806) - MIT; RLCR (reinforcement learning with calibration rewards), they add calibration term and help model verbalize its confidence (claims it generalizes to out of domain tasks also, need to reread that part)

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
- [Reasoning Models Can Be Effective Without Thinking](https://arxiv.org/abs/2504.09858) - basically skip the thinking step in a thinking model
- [RARE: Retrieval-Augmented Reasoning Modeling](https://arxiv.org/abs/2503.23513) - no code/examples yet in preprint
- [Efficient Reasoning Models: A Survey](https://arxiv.org/abs/2504.10903)
- [A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce](https://arxiv.org/abs/2504.11343)
- [Climbing the Ladder of Reasoning: What LLMs Can-and Still Can't-Solve after SFT?](https://arxiv.org/abs/2504.11741)
- [SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution](https://arxiv.org/abs/2502.18449)
- [Language Models can Self-Improve at State-Value Estimation for Better Search](https://arxiv.org/abs/2503.02878)
- [SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks](https://arxiv.org/abs/2503.15478) - not directly about a reasoning model but nice RL and task area, collaborating with reasoning system etc.
- [Learning to Reason under Off-Policy Guidance](https://arxiv.org/abs/2504.14945)
- [FlowReasoner: Reinforcing Query-Level Meta-Agents](https://arxiv.org/abs/2504.15257)
- [ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958) - not directly reasoning model, but GRPO for tool use; general "tool use RL" area interesting
- [OTC: Optimal Tool Calls via Reinforcement Learning](https://arxiv.org/abs/2504.14870) - see above, "tool use RL"
- [Learning Adaptive Parallel Reasoning with Language Models](https://arxiv.org/abs/2504.15466)
- [Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning](https://arxiv.org/abs/2504.15275)
- [Tina: Tiny Reasoning Models via LoRA](https://arxiv.org/abs/2504.15777)
- [Process Reward Models That Think](https://arxiv.org/abs/2504.16828)
- [Offline Reinforcement Learning for LLM Multi-Step Reasoning](https://arxiv.org/abs/2412.16145)
- [Zero-shot Robotic Manipulation with Language-guided Instruction and Formal Task Planning](https://arxiv.org/abs/2501.15214) - not directly about reasoning models, it's symbolic reasoning for robotics and the "reasoning plan" general area; TODO: search for reasoning models in robotics and create Topic for it
- [SPC: Evolving Self-Play Critic via Adversarial Games for LLM Reasoning](https://arxiv.org/abs/2504.19162)
- [Toward Evaluative Thinking: Meta Policy Optimization with Evolving Reward Models](https://arxiv.org/abs/2504.20157)
- [Phi-4-reasoning Technical Report](https://arxiv.org/abs/2504.21318) - Microsoft
- [DeepCritic: Deliberate Critique with Large Language Models](https://arxiv.org/abs/2505.00662)
- [reWordBench: Benchmarking and Improving the Robustness of Reward Models with Transformed Inputs](https://arxiv.org/abs/2503.11751)
- [Toward Evaluative Thinking: Meta Policy Optimization with Evolving Reward Models](https://arxiv.org/abs/2504.20157) - nice area, but code not available yet it seems
- [Llama-Nemotron: Efficient Reasoning Models](https://arxiv.org/abs/2505.00949) - Nvidia, goes into technical details and implementation
- [Optimizing Chain-of-Thought Reasoners via Gradient Variance Minimization in Rejection Sampling and RL](https://arxiv.org/abs/2505.02391)
- [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/abs/2505.04588)
- [Scalable Chain of Thoughts via Elastic Reasoning](https://arxiv.org/abs/2505.05315)
- [LIMR: Less is More for RL Scaling](https://arxiv.org/abs/2502.11886)
- [The First Few Tokens Are All You Need: An Efficient and Effective Unsupervised Prefix Fine-Tuning Method for Reasoning Models](https://arxiv.org/abs/2503.02875)
- [Retro-Search: Exploring Untaken Paths for Deeper and Efficient Reasoning](https://arxiv.org/abs/2504.04383)
- [MiMo: Unlocking the Reasoning Potential of Language Model -- From Pretraining to Posttraining](https://arxiv.org/abs/2505.07608) - Xiaomi, claims 7B performance >= to o1-mini
- [Learning from Peers in Reasoning Models](https://arxiv.org/abs/2505.07787)
- [AM-Thinking-v1: Advancing the Frontier of Reasoning at 32B Scale](https://arxiv.org/abs/2505.08311)
- [Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures](https://arxiv.org/abs/2505.09343) - DeepSeek, not directly about reasoning but nice technical comments
- [Kalman Filter Enhanced GRPO for Reinforcement Learning-Based Language Model Reasoning](https://arxiv.org/abs/2505.07527)
- [The CoT Encyclopedia: Analyzing, Predicting, and Controlling how a Reasoning Model will Think](https://arxiv.org/abs/2505.10185)
- [ImagineBench: Evaluating Reinforcement Learning with Large Language Model Rollouts](https://arxiv.org/abs/2505.10010) - nice approach, added since may be useful for reasoning models and RL
- [How Difficulty-Aware Staged Reinforcement Learning Enhances LLMs' Reasoning Capabilities: A Preliminary Experimental Study](https://arxiv.org/abs/2504.00829)
- [Beyond 'Aha!': Toward Systematic Meta-Abilities Alignment in Large Reasoning Models](https://arxiv.org/abs/2505.10554)
- [AdaptThink: Reasoning Models Can Learn When to Think](https://arxiv.org/abs/2505.13417)
- [Optimizing Anytime Reasoning via Budget Relative Policy Optimization](https://arxiv.org/abs/2505.13438)
- [Reward Reasoning Model](https://arxiv.org/abs/2505.14674)
- [General-Reasoner: Advancing LLM Reasoning Across All Domains](https://arxiv.org/abs/2505.14652)
- [Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning](https://arxiv.org/abs/2505.16410)
- [Training Step-Level Reasoning Verifiers with Formal Verification Tools](https://arxiv.org/abs/2505.15960)
- [The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning](https://arxiv.org/abs/2505.15134) - they claim a training-free approach that uses inference-time modifications to logits that boosts "reasoning" dataset performance, amongst other things
- [QwenLong-L1: Towards Long-Context Large Reasoning Models with Reinforcement Learning](https://arxiv.org/abs/2505.17667)
- [VeriThinker: Learning to Verify Makes Reasoning Model Efficient](https://arxiv.org/abs/2505.17941)
- [ARM: Adaptive Reasoning Model](https://arxiv.org/abs/2505.20258)
- [ShorterBetter: Guiding Reasoning Models to Find Optimal Inference Length for Efficient Reasoning](https://arxiv.org/abs/2504.21370)
- [SynLogic: Synthesizing Verifiable Reasoning Data at Scale for Learning Logical Reasoning and Beyond](https://arxiv.org/abs/2505.19641)
- [rStar-Coder: Scaling Competitive Code Reasoning with a Large-Scale Verified Dataset](https://arxiv.org/abs/2505.21297) - Microsoft, but code/dataset isn't available yet
- [RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning](https://arxiv.org/abs/2505.15034)
- [Skywork Open Reasoner 1 Technical Report](https://arxiv.org/abs/2505.22312)
- [Are Reasoning Models More Prone to Hallucination?](https://arxiv.org/abs/2505.23646)
- [LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling](https://arxiv.org/abs/2505.19187)
- [R2R: Efficiently Navigating Divergent Reasoning Paths with Small-Large Model Token Routing](https://arxiv.org/abs/2505.21600) - Tsinghua, cool test-time scaling approach using SLM and larger LLM
- [The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models](https://arxiv.org/abs/2505.22617) - theoretical paper, need to reread carefully
- [Pitfalls of Rule- and Model-based Verifiers -- A Case Study on Mathematical Reasoning](https://arxiv.org/abs/2505.22203) - they study the pros and cons of different verifiers when doing RL; the focus in on mathematical dataset/examples but it seems broadly applicable to RLVR
- [System-1.5 Reasoning: Traversal in Language and Latent Spaces with Dynamic Shortcuts](https://arxiv.org/abs/2505.18962) - CoCoNut paper related, see also review paper Reasoning Beyond Language linked below
- [Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](https://arxiv.org/abs/2506.01939)
- [AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning](https://arxiv.org/abs/2505.24298)
- [Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models](https://arxiv.org/abs/2506.01413)
- [R1-Code-Interpreter: Training LLMs to Reason with Code via Supervised and Reinforcement Learning](https://arxiv.org/abs/2505.21668)
- [RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling](https://arxiv.org/abs/2506.08672)
- [Multiverse: Your Language Models Secretly Decide How to Parallelize and Merge Generation](https://arxiv.org/abs/2506.09991)
- [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585)
- [AceReason-Nemotron 1.1: Advancing Math and Code Reasoning through SFT and RL Synergy](https://arxiv.org/abs/2506.13284) - Nvidia; nice improvements since 1.0 model (still 7B based on Qwen 2.5)
- [Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs](https://arxiv.org/abs/2506.14245)
- [ProtoReasoning: Prototypes as the Foundation for Generalizable Reasoning in LLMs](https://arxiv.org/abs/2506.15211) -  ByteDance; "... we propose ProtoReasoning, a framework that enhances the reasoning ability of LLMs by leveraging scalable and verifiable prototypical representations (Prolog for logical reasoning, PDDL for planning) ... Significantly, our ablation studies confirm that learning in prototype space also demonstrates enhanced generalization to structurally similar problems compared to training solely on natural language representations, validating our hypothesis that reasoning prototypes serve as the foundation for generalizable reasoning in large language models."
- [Revisiting Reinforcement Learning for LLM Reasoning from A Cross-Domain Perspective](https://arxiv.org/abs/2506.14965) - links to nice Github pages also [https://guru-reasoning.github.io/](https://guru-reasoning.github.io/) and [https://github.com/LLM360/Reasoning360](https://github.com/LLM360/Reasoning360)
- [Reward-SQL: Boosting Text-to-SQL via Stepwise Reasoning and Process-Supervised Rewards](https://arxiv.org/abs/2505.04671)
- [ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs](https://arxiv.org/abs/2506.18896)
- [Robust Reward Modeling via Causal Rubrics](https://arxiv.org/abs/2506.16507) - Google DeepMind; "We introduce Crome (Causally Robust Reward Modeling), a novel framework grounded in an explicit causal model designed to mitigate reward hacking.", has some results on reasoning benchmarks that seem good
- [OctoThinker: Mid-training Incentivizes Reinforcement Learning Scaling](https://arxiv.org/abs/2506.20512)
- [Reinforcement Learning Teachers of Test Time Scaling](https://arxiv.org/abs/2506.08388) - Sakana AI; Github [https://github.com/SakanaAI/RLT](https://github.com/SakanaAI/RLT) and blog [https://sakana.ai/rlt/](https://sakana.ai/rlt/)
- [Fractional Reasoning via Latent Steering Vectors Improves Inference Time Compute](https://arxiv.org/abs/2506.15882)
- [SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2506.24119)
- [Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning](https://arxiv.org/abs/2507.00432) - "Math reasoning has become the poster child of progress in large language models (LLMs), with new models rapidly surpassing human-level performance on benchmarks like MATH and AIME. But as math leaderboards improve week by week, it is worth asking: do these gains reflect broader problem-solving ability or just narrow overfitting? To answer this question, we evaluate over 20 open-weight reasoning-tuned models across a broad suite of tasks, including math, scientific QA, agent planning, coding, and standard instruction-following. We surprisingly find that most models that succeed in math fail to transfer their gains to other domains."
- [Fast and Simplex: 2-Simplicial Attention in Triton](https://arxiv.org/abs/2507.02754) - Meta; this is mainly an architecture/technical paper, but the claims are related specifically to improving performance on reasoning tasks - not clear to me yet, need to take notes carefully
- [Pre-Trained Policy Discriminators are General Reward Models](https://arxiv.org/abs/2507.05197) - not strictly about reasoning models but applicable and relevant
- [R1-RE: Cross-Domain Relationship Extraction with RLVR](https://arxiv.org/abs/2507.04642) - nice, classic NLP task; claim 7B model reaches 4o performance
- [Decoder-Hybrid-Decoder Architecture for Efficient Reasoning with Long Generation](https://arxiv.org/abs/2507.06607) - Microsoft, this is the technical paper for the Phi 4 mini flash reasoning model (3.8B) with some nice architecture ideas. The blogpost is also here: [Reasoning reimagined: Introducing Phi-4-mini-flash-reasoning](https://azure.microsoft.com/en-us/blog/reasoning-reimagined-introducing-phi-4-mini-flash-reasoning)
- [First Return, Entropy-Eliciting Explore](https://arxiv.org/abs/2507.07017) - ByteDance
- [Test-Time Scaling with Reflective Generative Model](https://arxiv.org/abs/2507.01951)
- [REST: Stress Testing Large Reasoning Models by Asking Multiple Problems at Once](https://arxiv.org/abs/2507.10541)
- [Logit Arithmetic Elicits Long Reasoning Capabilities Without Training](https://www.arxiv.org/abs/2507.12759) - reminds me a bit of The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning (see link above)
- [Beyond Context Limits: Subconscious Threads for Long-Horizon Reasoning](https://arxiv.org/abs/2507.16784)
- [MegaScience: Pushing the Frontiers of Post-Training Datasets for Science Reasoning](https://arxiv.org/abs/2507.16812) - has nice explanation of data filtering and processing also
- [Can One Domain Help Others? A Data-Centric Study on Multi-Domain Reasoning via Reinforcement Learning](https://arxiv.org/abs/2507.17512)
- [LAPO: Internalizing Reasoning Efficiency via Length-Adaptive Policy Optimization](https://arxiv.org/abs/2507.15758)
- [Re:Form -- Reducing Human Priors in Scalable Formal Software Verification with RL in LLMs: A Preliminary Study on Dafny](https://arxiv.org/abs/2507.16331) - formal methods
- [KV Cache Steering for Inducing Reasoning in Small Language Models](https://arxiv.org/abs/2507.08799) - it is a one-shot method, seems cool: " Our approach leverages GPT-4o-generated reasoning traces to construct steering vectors that shift model behavior toward more explicit, multi-step reasoning without fine-tuning or prompt modifications."; examples on the Github [https://github.com/MaxBelitsky/cache-steering](https://github.com/MaxBelitsky/cache-steering)
- [UloRL:An Ultra-Long Output Reinforcement Learning Approach for Advancing Large Language Models' Reasoning Abilities](https://arxiv.org/abs/2507.19766) - Tencent, nice results/performance; code is available
- [Beyond the Trade-off: Self-Supervised Reinforcement Learning for Reasoning Models' Instruction Following](https://arxiv.org/abs/2508.02150)
- [CompassVerifier: A Unified and Robust Verifier for LLMs Evaluation and Outcome Reward](https://arxiv.org/abs/2508.03686)

#### Topic - Verifier-free RL and approaches without External Rewards

**Emerging topic, seen a few papers around this theme recently.**

- [Learning to Reason without External Rewards](https://arxiv.org/abs/2505.19590) - "We propose Intuitor, an RLIF method that uses a model's own confidence, termed self-certainty, as its sole reward signal. Intuitor replaces external rewards in Group Relative Policy Optimization (GRPO) with self-certainty scores, enabling fully unsupervised learning. Experiments demonstrate that Intuitor matches GRPO's performance on mathematical benchmarks while achieving superior generalization to out-of-domain tasks like code generation, without requiring gold solutions or test cases."
- [NOVER: Incentive Training for Language Models via Verifier-Free Reinforcement Learning](https://arxiv.org/abs/2505.16022)
- [Reinforcing General Reasoning without Verifiers](https://arxiv.org/abs/2505.21493)
- [RLPR: Extrapolating RLVR to General Domains without Verifiers](https://arxiv.org/abs/2506.18254)
- [The Invisible Leash: Why RLVR May Not Escape Its Origin](https://arxiv.org/abs/2507.14843)
- [Stabilizing Knowledge, Promoting Reasoning: Dual-Token Constraints for RLVR](https://arxiv.org/abs/2507.15778)

#### Topic - Surveys and reviews

- [Towards Large Reasoning Models: A Survey of Reinforced Reasoning with Large Language Models](https://arxiv.org/abs/2501.09686)
- [Reinforcement Learning Enhanced LLMs: A Survey](https://arxiv.org/abs/2412.10400)
- [From System 1 to System 2: A Survey of Reasoning Large Language Models](https://arxiv.org/abs/2502.17419)
- [What, How, Where, and How Well? A Survey on Test-Time Scaling in Large Language Models](https://arxiv.org/abs/2503.24235) - section 4.1 is on reasoning specifically but general paper is nicely organized and relevant
- [Knowledge Augmented Complex Problem Solving with Large Language Models: A Survey](https://arxiv.org/abs/2505.03418) - Zhejiang, mainly reasoning related topics
- [Perception, Reason, Think, and Plan: A Survey on Large Multimodal Reasoning Models](https://arxiv.org/abs/2505.04921)
- [Sailing AI by the Stars: A Survey of Learning from Rewards in Post-Training and Test-Time Scaling of Large Language Models](https://arxiv.org/abs/2505.02686) - nice related topics in general about learning from rewards, also has nice Github [https://github.com/bobxwu/learning-from-rewards-llm-papers](https://github.com/bobxwu/learning-from-rewards-llm-papers)
- [Reasoning Beyond Language: A Comprehensive Survey on Latent Chain-of-Thought Reasoning](https://arxiv.org/abs/2505.16782) - CoCoNut type experiments on latent space reasoning etc.
- [A Survey on Latent Reasoning](https://arxiv.org/abs/2507.06203) - " We begin by examining the foundational role of neural network layers as the computational substrate for reasoning, highlighting how hierarchical representations support complex transformations. Next, we explore diverse latent reasoning methodologies, including activation-based recurrence, hidden state propagation, and fine-tuning strategies that compress or internalize explicit reasoning traces. Finally, we discuss advanced paradigms such as infinite-depth latent reasoning via masked diffusion models, which enable globally consistent and reversible reasoning processes."

#### Topic - Tool Use/Function Calling

- [FunReason: Enhancing Large Language Models' Function Calling via Self-Refinement Multiscale Loss and Automated Data Refinement](https://arxiv.org/abs/2505.20192) - general methodology

#### Topic - Reasoning + Agents

- [ATLaS: Agent Tuning via Learning Critical Steps](https://arxiv.org/abs/2503.02197)
- [Agent models: Internalizing Chain-of-Action Generation into Reasoning models](https://arxiv.org/abs/2503.06580)
- [The Danger of Overthinking: Examining the Reasoning-Action Dilemma in Agentic Tasks](https://arxiv.org/abs/2502.08235)
- [Distilling LLM Agent into Small Models with Retrieval and Code Tools](https://arxiv.org/abs/2505.17612) - nice, not stricty about agents but more about distillation; read this comment on HF for quick summary [https://huggingface.co/posts/m-ric/682683815641001](https://huggingface.co/posts/m-ric/682683815641001); "when trying to distil reasoning capability from a strong LLM ("teacher") into a smaller one ("student"), it's much better to use Agent traces than CoT traces."
- [Thinking vs. Doing: Agents that Reason by Scaling Test-Time Interaction](https://arxiv.org/abs/2506.07976) - nice paper

#### Topic - Information retrieval and search

- [Learning More Effective Representations for Dense Retrieval through Deliberate Thinking Before Search](https://arxiv.org/abs/2502.12974)
- [O1 Embedder: Let Retrievers Think Before Action](https://arxiv.org/abs/2502.07555)
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516)
- [Open Deep Search: Democratizing Search with Open-source Reasoning Agents](https://arxiv.org/abs/2503.20201)
- [Distillation and Refinement of Reasoning in Small Language Models for Document Re-ranking](https://arxiv.org/abs/2504.03947) - uses BRIGHT benchmark
- [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160) - deep research, does have a reasoning type model involved for planning stage
- [DeepRetrieval: Hacking Real Search Engines and Retrievers with Large Language Models via Reinforcement Learning](https://arxiv.org/abs/2503.00223) - results seem good
- [ReasonIR: Training Retrievers for Reasoning Tasks](https://arxiv.org/abs/2504.20595) - uses BRIGHT benchmark
- [An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents](https://arxiv.org/abs/2505.15117) - follow-up to Search-R1 paper
- [FREESON: Retriever-Free Retrieval-Augmented Reasoning via Corpus-Traversing MCTS](https://arxiv.org/abs/2505.16409)
- [ConvSearch-R1: Enhancing Query Reformulation for Conversational Search with Reasoning via Reinforcement Learning](https://arxiv.org/abs/2505.15776)
- [VRAG-RL: Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning with Reinforcement Learning](https://arxiv.org/abs/2505.22019) - multimodal
- [From Token to Action: State Machine Reasoning to Mitigate Overthinking in Information Retrieval](https://arxiv.org/abs/2505.23059) - uses BRIGHT benchmark, defines a refine and rerank operation that they use to cut down token use compared to CoT
- [R2ec : Towards Large Recommender Models with Reasoning](https://arxiv.org/abs/2505.16994) - use reasoning for recommendation systems
- [Decoupled Planning and Execution: A Hierarchical Reasoning Framework for Deep Search](https://arxiv.org/abs/2507.02652)

#### Topic - X of Thoughts variants

- [Path-of-Thoughts: Extracting and Following Paths for Robust Relational Reasoning with Large Language Models](https://arxiv.org/abs/2412.17963)
- [Forest-of-Thought: Scaling Test-Time Compute for Enhancing LLM Reasoning](https://arxiv.org/abs/2412.09078)
- [AdaCoT: Pareto-Optimal Adaptive Chain-of-Thought Triggering via Reinforcement Learning](https://arxiv.org/abs/2505.11896)

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
- [SpecReason: Fast and Accurate Inference-Time Compute via Speculative Reasoning](https://arxiv.org/abs/2504.07891)
- [SeerAttention-R: Sparse Attention Adaptation for Long Reasoning](https://arxiv.org/abs/2506.08889) - Microsoft Research; uses [https://github.com/tile-ai/tilelang](https://github.com/tile-ai/tilelang) that I need to learn about also O_o
- [R-KV: Redundancy-aware KV Cache Compression for Training-Free Reasoning Models Acceleration](https://arxiv.org/abs/2505.24133)
- [Inference-Time Hyper-Scaling with KV Cache Compression](https://arxiv.org/abs/2506.05345)

#### Topic - Ensembling, Boosting, Stacking etc.

- [Ensembling Large Language Models with Process Reward-Guided Tree Search for Better Complex Reasoning](https://arxiv.org/abs/2412.15797)

#### Topic - Evaluation of reasoning

- [Are Your LLMs Capable of Stable Reasoning?](https://arxiv.org/abs/2412.13147)
- [The Jumping Reasoning Curve? Tracking the Evolution of Reasoning Performance in GPT-\[n\] and o-\[n\] Models on Multimodal Puzzles](https://arxiv.org/abs/2502.01081)
- [xVerify: Efficient Answer Verifier for Reasoning Model Evaluations](https://arxiv.org/abs/2504.10481)

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
- [Kimina-Prover Preview: Towards Large Formal Reasoning Models with Reinforcement Learning](https://arxiv.org/abs/2504.11354) - Lean
- [AIMO-2 Winning Solution: Building State-of-the-Art Mathematical Reasoning Models with OpenMathReasoning dataset](https://arxiv.org/abs/2504.16891) - Nvidia
- [Phi-4-Mini-Reasoning: Exploring the Limits of Small Reasoning Language Models in Math](https://arxiv.org/abs/2504.21233) - Microsoft technical report
- [FormalMATH: Benchmarking Formal Mathematical Reasoning of Large Language Models](https://arxiv.org/abs/2505.02735) - benchmark but technical report is nice, also there is a Github at [https://spherelab.ai/FormalMATH/](https://spherelab.ai/FormalMATH/)
- [CombiBench: Benchmarking LLM Capability for Combinatorial Mathematics](https://arxiv.org/abs/2505.03171) - finally some combinatorics :) seems like a difficult area, model performance on benchmark is quite low even with Kimina Prover
- [MPS-Prover: Advancing Stepwise Theorem Proving by Multi-Perspective Search and Data Curation](https://arxiv.org/abs/2505.10962)
- [DeepTheorem: Advancing LLM Reasoning for Theorem Proving Through Natural Language and Reinforcement Learning](https://arxiv.org/abs/2505.23754)
- [Mathesis: Towards Formal Theorem Proving from Natural Languages](https://arxiv.org/abs/2506.07047)
- [Bourbaki: Self-Generated and Goal-Conditioned MDPs for Theorem Proving](https://arxiv.org/abs/2507.02726)
- [Towards Solving More Challenging IMO Problems via Decoupled Reasoning and Proving](https://arxiv.org/abs/2507.06804) - Tencent, nice approach
- [MiroMind-M1: An Open-Source Advancement in Mathematical Reasoning via Context-Aware Multi-Stage Policy Optimization](https://arxiv.org/abs/2507.14683)
- [Seed-Prover: Deep and Broad Reasoning for Automated Theorem Proving](https://arxiv.org/abs/2507.23726) - ByteDance; 5/6 problems on IMO 2025

#### Topic - Coding

- [ACECODER: Acing Coder RL via Automated Test-Case Synthesis](https://arxiv.org/abs/2502.01718)
- [S*: Test Time Scaling for Code Generation](https://arxiv.org/abs/2502.14382)
- [Learning to Solve and Verify: A Self-Play Framework for Code and Test Generation](https://arxiv.org/abs/2502.14948) - not directly about reasoning but nice approach
- [Think Like Human Developers: Harnessing Community Knowledge for Structured Code Reasoning](https://arxiv.org/abs/2503.14838)
- [LiveCodeBench Pro: How Do Olympiad Medalists Judge LLMs in Competitive Programming?](https://arxiv.org/abs/2506.11928)

#### Topic - Biology

- [BioProBench: Comprehensive Dataset and Benchmark in Biological Protocol Understanding and Reasoning](https://arxiv.org/abs/2505.07889) - really cool area
- [BioReason: Incentivizing Multimodal Biological Reasoning within a DNA-LLM Model](https://arxiv.org/abs/2505.23579)

#### Topic - Other specialized reasoning areas (medical, legal, etc.)

- [ChemAgent: Self-updating Library in Large Language Models Improves Chemical Reasoning](https://arxiv.org/abs/2501.06590)
- [MedS3: Towards Medical Small Language Models with Self-Evolved Slow Thinking](https://arxiv.org/abs/2501.12051)
- [MedAgentsBench: Benchmarking Thinking Models and Agent Frameworks for Complex Medical Reasoning](https://arxiv.org/abs/2503.07459)
- [Fin-R1: A Large Language Model for Financial Reasoning through Reinforcement Learning](https://arxiv.org/abs/2503.16252)
- [m1: Unleash the Potential of Test-Time Scaling for Medical Reasoning with Large Language Models](https://arxiv.org/abs/2504.00869)
- [DianJin-R1: Evaluating and Enhancing Financial Reasoning in Large Language Models](https://arxiv.org/abs/2504.15716) - Qwen team
- [MedCaseReasoning: Evaluating and learning diagnostic reasoning from clinical case reports](https://arxiv.org/abs/2505.11733)
- [CodeV-R1: Reasoning-Enhanced Verilog Generation](https://arxiv.org/abs/2505.24183) - hardware design language HDL
- [MedAgentGym: Training LLM Agents for Code-Based Medical Reasoning at Scale](https://arxiv.org/abs/2506.04405)
- [Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning](https://arxiv.org/abs/2506.07044)
- [ReasonMed: A 370K Multi-Agent Generated Dataset for Advancing Medical Reasoning](https://arxiv.org/abs/2506.09513)
- [Enhancing Step-by-Step and Verifiable Medical Reasoning in MLLMs](https://arxiv.org/abs/2506.16962)
- [An Agentic System for Rare Disease Diagnosis with Traceable Reasoning](https://arxiv.org/abs/2506.20430)
- [ChemDFM-R: An Chemical Reasoner LLM Enhanced with Atomized Chemical Knowledge](https://arxiv.org/abs/2507.21990)

#### Topic - Multimodal

TODO: expand with separate tags

- [Multimodal Chain-of-Thought Reasoning: A Comprehensive Survey](https://arxiv.org/abs/2503.12605)
- [https://embodied-reasoner.github.io/](https://embodied-reasoner.github.io/)
- [Skywork R1V: Pioneering Multimodal Reasoning with Chain-of-Thought](https://arxiv.org/abs/2504.05599)
- [Bring Reason to Vision: Understanding Perception and Reasoning through Model Merging](https://arxiv.org/abs/2505.05464) - cool, claims transfering reasoning ability from LLM to VLM
- [Active-O3: Empowering Multimodal Large Language Models with Active Perception via GRPO](https://arxiv.org/abs/2505.21457) - Zhejiang and Ant Group, really cool
- [Unsupervised Post-Training for Multi-Modal LLM Reasoning via GRPO](https://arxiv.org/abs/2505.22453)
- [VideoDeepResearch: Long Video Understanding With Agentic Tool Using](https://arxiv.org/abs/2506.10821) - "Our approach relies solely on a text-only large reasoning model (LRM) combined with a modular multi-modal toolkit, including multimodal retrievers and visual perceivers, all of which are readily available in practice. For each LVU task, the system formulates a problem-solving strategy through reasoning, while selectively accessing and utilizing essential video content via tool using."
- [MiCo: Multi-image Contrast for Reinforcement Visual Reasoning](https://arxiv.org/abs/2506.22434) - "Without relying on any human-annotated question-answer pairs, our method achieves significant improvements on multi-image reasoning benchmarks and shows strong performance on general vision tasks."
- [Fine-Grained Preference Optimization Improves Spatial Reasoning in VLMs](https://arxiv.org/abs/2506.21656) - spatial reasoning
- [Scaling RL to Long Videos](https://arxiv.org/abs/2507.07966) - "We address the unique challenges of long video reasoning by integrating three critical components: (1) a large-scale dataset, LongVideo-Reason, comprising 52K long video QA pairs with high-quality reasoning annotations across diverse domains such as sports, games, and vlogs; (2) a two-stage training pipeline that extends VLMs with chain-of-thought supervised fine-tuning (CoT-SFT) and reinforcement learning (RL); and (3) a training infrastructure for long video RL, named Multi-modal Reinforcement Sequence Parallelism (MR-SP), which incorporates sequence parallelism and a vLLM-based engine tailored for long video, using cached video embeddings for efficient rollout and prefilling."
- [Scientists' First Exam: Probing Cognitive Abilities of MLLM via Perception, Understanding, and Reasoning](https://arxiv.org/abs/2506.10521) - cool paper and benchmark
- [MindJourney: Test-Time Scaling with World Models for Spatial Reasoning](https://arxiv.org/abs/2507.12508)
- [M2-Reasoning: Empowering MLLMs with Unified General and Spatial Reasoning](https://arxiv.org/abs/2507.08306) - Ant group
- [Zebra-CoT: A Dataset for Interleaved Vision Language Reasoning](https://arxiv.org/abs/2507.16746) - cool dataset, using a kind of "sketchpad" while reasoning
- [ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning](https://arxiv.org/abs/2507.16815) - NVIDIA; robotics related also, nice VLA approach
- [3D-R1: Enhancing Reasoning in 3D VLMs for Unified Scene Understanding](https://arxiv.org/abs/2507.23478) - not sure if I understood appendix correctly but it seems the model dynamically moves around the 3D scene to get better perspectives to answer queries; cool approach if so, need to reread the details
- [InstructVLA: Vision-Language-Action Instruction Tuning from Understanding to Manipulation](https://arxiv.org/abs/2507.17520) - "Additionally, InstructVLA surpasses baseline VLMs on multimodal tasks and exhibits inference-time scaling by leveraging textual reasoning to boost manipulation performance in both simulated and real-world settings."

#### Topic - Robotics

- [RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics](https://arxiv.org/abs/2506.04308)

#### Topic - Adjacent subjects, maybe related

Personal reading notes/papers that gave me some reasoning-related ideas

- [Answer Set Networks: Casting Answer Set Programming into Deep Learning](https://arxiv.org/abs/2412.14814)
- [UQE: A Query Engine for Unstructured Databases](https://arxiv.org/abs/2407.09522)
- [LTNtorch: PyTorch Implementation of Logic Tensor Networks](https://arxiv.org/abs/2409.16045)
- [SymBa: Symbolic Backward Chaining for Structured Natural Language Reasoning](https://arxiv.org/abs/2402.12806)
- [Shifting Long-Context LLMs Research from Input to Output](https://arxiv.org/abs/2503.04723) - generating large reasoning chains
- [BlendRL: A Framework for Merging Symbolic and Neural Policy Learning](https://arxiv.org/abs/2410.11689) - ICLR 2025
- [Critical Thinking: Which Kinds of Complexity Govern Optimal Reasoning Length?](https://arxiv.org/abs/2504.01935) - more theoretical but maybe interesting
- [Pantograph: A Machine-to-Machine Interaction Interface for Advanced Theorem Proving, High Level Reasoning, and Data Extraction in Lean 4](https://arxiv.org/abs/2410.16429) - Lean, mathematics

---

## Datasets

- [https://huggingface.co/datasets/open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) - OpenR1-Math-220k is a large-scale dataset for mathematical reasoning. It consists of 220k math problems with two to four reasoning traces generated by DeepSeek R1 for problems from NuminaMath 1.5. The traces were verified using Math Verify for most samples and Llama-3.3-70B-Instruct as a judge for 12% of the samples, and each problem contains at least one reasoning trace with a correct answer.
- [https://huggingface.co/collections/open-r1/reasoning-datasets-67980cac6e816a0eda98c678](https://huggingface.co/collections/open-r1/reasoning-datasets-67980cac6e816a0eda98c678) - Datasets with reasoning traces for math and code released by the community
- [DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset for Advancing Reasoning](https://arxiv.org/abs/2504.11456) - DeepMath-103K, a new, large-scale dataset comprising approximately 103K mathematical problems, specifically designed to train advanced reasoning models via RL.

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
- [Nondeterministic Polynomial-time Problem Challenge: An Ever-Scaling Reasoning Benchmark for LLMs](https://arxiv.org/abs/2504.11239) - cool idea but no Github/access to benchmark yet? 
- [CipherBank: Exploring the Boundary of LLM Reasoning Capabilities through Cryptography Challenges](https://arxiv.org/abs/2504.19093)
- [DeepMath-Creative: A Benchmark for Evaluating Mathematical Creativity of Large Language Models](https://arxiv.org/abs/2505.08744) - nice area, more creative reasoning about mathematical problems and deduction
- [R2MED: A Benchmark for Reasoning-Driven Medical Retrieval](https://arxiv.org/abs/2505.14558)

---

## Blogs and articles

- [HKUST - Simple Reinforcement Learning for Reasoning](https://hkust-nlp.notion.site/simplerl-reason)
- [OpenAI - Deliberative alignment: reasoning enables safer language models](https://openai.com/index/deliberative-alignment/)
- [DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) - main project page is [https://agentica-project.com/](https://agentica-project.com/), HuggingFace page is [https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview)
- [Open R1: Update #3](https://huggingface.co/blog/open-r1/update-3) - HuggingFace Open-R1 update on Codeforces and IOI 24 Informatics Olympiads, datasets etc.
- [Understanding R1-Zero-Like Training: A Critical Perspective](https://github.com/sail-sg/understand-r1-zero)
- [YouTube - Nathan Lambert - GRPO's new variants and implementation secrets](https://www.youtube.com/watch?v=amrJDwMUFNs)
- [Reasoning models don't always say what they think](https://www.anthropic.com/research/reasoning-models-dont-say-think) - nice Anthropic blog, basically they take a reasoning model and place hints to correct answer in the input prompts (during eval), then measure whether model mentions it used the hint to reach its final answer etc.
- [DeepCoder: A Fully Open-Source 14B Coder at O3-mini Level](https://www.together.ai/blog/deepcoder) - Through a joint collaboration between the Agentica team and Together AI, we release DeepCoder-14B-Preview, a code reasoning model finetuned from Deepseek-R1-Distilled-Qwen-14B via distributed RL. It achieves an impressive 60.6% Pass@1 accuracy on LiveCodeBench (+8% improvement), matching the performance of o3-mini-2025-01-031 (Low) and o1-2024-12-17 with just 14B parameters. We’ve open-sourced our dataset, code, training logs, and systems optimizations for everyone to progress on scaling and accelerating intelligence with RL.
- [Shunyu Yao - The Second Half](https://ysymyth.github.io/The-Second-Half/) - awesome blog on status of reinforcement learning as of April 2025 by OpenAI researcher
- [ServiceNow Reseach - PipelineRL](https://huggingface.co/blog/ServiceNow/pipelinerl) - We are excited to open-source PipelineRL, an experimental RL implementation that tackles a fundamental challenge in large-scale Reinforcement Learning with LLMs: the trade-off between inference throughput and on-policy data collection. PipelineRL's key innovation is inflight weight updates during RL training. This allows PipelineRL to achieve constantly high inference throughput and minimize the lag between the weights used for rollouts and the most recently updated model weights. The result: fast and stable RL training for large language models.
- [OpenAI's o3: Over-optimization is back and weirder than ever](https://www.interconnects.ai/p/openais-o3-over-optimization-is-back) - Nathan Lambert blog
- [Reinforcement learning with random rewards actually works with Qwen 2.5](https://www.interconnects.ai/p/reinforcement-learning-with-random) - Nathan Lambert blog, see new papers Topic about "Verifier-free RL and approaches without External Rewards" above in main section
- [Crafting a good (reasoning) model](https://www.interconnects.ai/p/crafting-a-good-reasoning-model) - Nathan Lambert talk video/blog notes
- [The rise of reasoning machines](https://www.interconnects.ai/p/the-rise-of-reasoning-machines) - Nathan Lambert blog
- [POLARIS: A POst-training recipe for scaling reinforcement Learning on Advanced ReasonIng modelS](https://honorable-payment-890.notion.site/POLARIS-A-POst-training-recipe-for-scaling-reinforcement-Learning-on-Advanced-ReasonIng-modelS-1dfa954ff7c38094923ec7772bf447a1) - "Our 4B model achieves an impressive 81.2% Pass@1 accuracy on AIME24 and 79.4% Pass@1 accuracy on AIME25, outperforming state-of-the-art commercial models like Claude-4-Opus, Grok-3-Beta, and o3-mini-high(2025/01/31) via scaling reinforcement learning on open-source data. On AIME25, POLARIS astonishingly achieves comparable performance to Qwen3-235B-A22B  while using less than 2% of its parameters and can be deployed on consumer-grade GPUs."
- [HuggingFace Spaces - Scaling Test Time Compute with Open Models](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute) - from December 2024 but nice presentation and introduction
- [Self-Adapting Language Models](https://jyopari.github.io/posts/seal) - "We demonstrate SEAL in two domains: (1) Knowledge Incorporation, where the model integrates new factual information by generating logical implications as synthetic data, and (2) Few-Shot Learning, where the model autonomously selects data augmentations and training hyperparameters to adapt to new abstract reasoning tasks."
- [DeepSWE: Training a Fully Open-sourced, State-of-the-Art Coding Agent by Scaling RL](https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-Art-Coding-Agent-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33) - "We introduce DeepSWE-Preview, a reasoning-enabled coding agent trained from Qwen3-32B with only reinforcement learning (RL). It achieves  an impressive 59.0% on SWE-Bench-Verified with test-time scaling, reaching SOTA for open-weight coding agents  (42.2% Pass@1, 71.0% Pass@16). DeepSWE is trained using rLLM, our framework for post-training language agents. We’ve open sourced everything—our dataset, code, training, and eval logs, for everyone to progress on scaling and improving agents with RL."
- [rLLM: A Framework for Post-Training Language Agents](https://pretty-radio-b75.notion.site/rLLM-A-Framework-for-Post-Training-Language-Agents-21b81902c146819db63cd98a54ba5f31) - nice, learned about this from blog about DeepSWE above; good perspective on going beyond simple reasoning models
- [Detecting misbehavior in frontier reasoning models](https://openai.com/index/chain-of-thought-monitoring/) - OpenAI
- [What comes next with reinforcement learning](https://www.interconnects.ai/p/what-comes-next-with-reinforcement) - Nathan Lambert blog
- [Kimina-Prover: Applying Test-time RL Search on Large Formal Reasoning Models](https://huggingface.co/blog/AI-MO/kimina-prover) - mathematics and theorem proving: "Our key innovations include: 1) Test-Time Reinforcement Learning Search: A trainable agentic proving framework that enables the model to recursively discover, combine and apply multiple lemmas to construct complex proofs, building on a novel lemma-enabled pattern. 2) Error-Fixing Capability: Kimina-Prover can read and interpret Lean’s error messages and propose targeted fixes, demonstrating significantly higher sample efficiency compared to regenerating proofs from scratch. These advancements enable Kimina-Prover to solve challenging mathematical problems and surpass prior methods. As shown in Figure 1, on the widely used miniF2F benchmark, Kimina-Prover achieves a state-of-the-art pass rate of 92.2%."
- [Bourbaki (7b): SOTA 7B Algorithms for Putnam Bench (Part I: Reasoning MDPs)](https://huggingface.co/blog/hba123/bourbaki7b)
- [Context Rot: How Increasing Input Tokens Impacts LLM Performance](https://research.trychroma.com/context-rot) - not specifically about reasoning models but interesting for complex problem solving in general and related areas though
- [Scaling Test Time Compute to Multi-Agent Civilizations: Noam Brown](https://www.latent.space/p/noam-brown) - awesome interview and blog (o1 etc.)
- [Interviewing Ross Taylor on the state of AI: Chinese open models, scaling reasoning, useful tools, and what comes next](https://www.interconnects.ai/p/interviewing-ross-taylor-on-the-state) - Nathan Lambert blog; nice topics and discussion especially in 2nd half of the interview

#### Workshops

- [Reasoning and Planning for Large Language Models, ICLR 2025, April 28 2025, Singapore](https://workshop-llm-reasoning-planning.github.io/)
- [XLLM @ ACL 2025 Shared Task-III: LLM for Structural Reasoning (LLM-SR)](https://xllms.github.io/LLMSR/) - saw several papers and submissions to this, e.g. [LLMSR@XLLM25: Less is More: Enhancing Structured Multi-Agent Reasoning via Quality-Guided Distillation](https://arxiv.org/abs/2504.16408) and [LLMSR@XLLM25: An Empirical Study of LLM for Structural Reasoning](https://arxiv.org/abs/2505.12328)

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
- [DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition](https://github.com/deepseek-ai/DeepSeek-Prover-V2) - see the paper also in the Github

#### Collections

- [Linked from paper https://arxiv.org/abs/2501.02497 - This repository contains the resources for Test-time Computing: from System-1 Thinking to System-2 Thinking](https://github.com/Dereck0602/Awesome_Test_Time_LLMs)
- [Curated collection of papers and resources on how to unlock the reasoning ability of LLMs and MLLMs.](https://github.com/atfortes/Awesome-LLM-Reasoning)
- [A curated list of language modeling researches for code (and other software engineering activities), plus related datasets.](https://github.com/codefuse-ai/Awesome-Code-LLM) - has related subjects around "code enhances reasoning" general area

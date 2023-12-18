# SLM (Small Language Model)

This represents my goal of training my own transformer-based model following [Andrej Karpathy's Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html),
specifically [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) and [nanoGPT](https://github.com/karpathy/nanoGPT).

> Note: This is not intended to be installed as a package!

- [SLM (Small Language Model)](#slm-small-language-model)
  - [Goals](#goals)
  - [Additional Experiments](#additional-experiments)
  - [Notes on Infrastructure](#notes-on-infrastructure)
  - [References](#references)
    - [Explainers](#explainers)
    - [Models](#models)
    - [Components](#components)

## Goals

1. Pretrain a GPT-style (Generative Pretrained Transformer) foundational model
   Per 'mamba' paper, use current SOTA architecture "rotary embedding, SwiGLU MLP, RMSNorm instead of LayerNorm, no linear bias, and higher learning rates"
2. Fine-tune the model for some task (TBD)
3. Evaluate responses and [use RLHF (Reinforcement Learning w/ Human Feedback) or DPO (Direct Preference Optimization) to align model with expected response quality](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives).
   This may require self-deploying an annotation tool like [doccano](https://github.com/doccano/doccano) or [Label Studio](https://labelstud.io/guide/get_started.html#Quick-start)
4. Deploy API endpoint
5. Deploy API endpoint that leverages RAG (Retrieval Augmented Generation) for improved grounding
6. [Fine-tune with Q/LoRA (Quantized Low-Rank Adaptation)](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)

## Additional Experiments

- Evaluate nonlinearities in style of [Karpathy](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)
  see: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202v1)
- Given scaling laws expectations [1](https://arxiv.org/abs/2001.08361), [2](https://blog.eleuther.ai/transformer-math/), [3](https://arxiv.org/abs/2203.15556), what is the optimal dataset and model params given my compute (RTX 3090)?
  Is it possible to reduce model compute requirements from full RTX 3090 --> CPU?
- Given easily-available training sets, how does `SLM` perform on bias and toxicity?
- Word2vec
  - find linear transformation to get (gender, royalty) hyperplane for `king - man + woman = queen`?
  - tok2vec - same process but with BPE (byte-pair encoding)? (is this ~ FastText?)
- BERT

## Notes on Infrastructure

I do most of my dev work on a Macbook Pro M1 with 16 GB RAM.  For training, I plan to use my PC which has 32 GB RAM and an RTX3090 (24 GB VRAM).
This may not be optimal given [Tim Dettmers' Jan 2023 investigation](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) but it is cheap (i.e., I already have it).
I intend to host the model, API Endpoints, and annotation tooling on my [self-hosted k3s cluster](https://github.com/ahgraber/homelab-gitops-k3s);
these nodes have an Intel i3-10100T and 32 GB RAM -- therefore, the model must be able to be run on CPU!

## References

### Explainers

- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [The Architecture of Today's LLM Applications](https://github.blog/2023-10-30-the-architecture-of-todays-llm-applications/)
- [Patterns for Building LLM-based Systems & Products](https://eugeneyan.com/writing/llm-patterns/)
- [Building LLM applications for production](https://huyenchip.com/2023/04/11/llm-engineering.html)
- [RLHF: Reinforcement Learning from Human Feedback](https://huyenchip.com/2023/05/02/rlhf.html)
- [SwiGLU: GLU Variants Improve Transformer](https://kikaben.com/swiglu-2020/)
- [Transformers from Scratch](https://e2eml.school/transformers.html)
- [Transformer Math 101](https://blog.eleuther.ai/transformer-math/)
- [The Transformer Family](https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/)
- [On the Dangers of Stochastic Parrots](https://dl.acm.org/doi/10.1145/3442188.3445922)

### Models

- Word2Vec (2013) [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- BERT (2018): [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- (2020) [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- Chinchilla (2022): [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- GPT (2018): [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- GPT-2 (2019): [Language Models are Unsupervised Multi-Task Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- GPT-3 (2020): [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- InstructGPT (2022): [Training Language Models to Follow Instructions](https://arxiv.org/abs/2203.02155)
- LLaMA (2023): [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- Llama 2 (2023): [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)

### Components

- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202v1)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [The Transformer Blueprint](https://deeprevision.github.io/posts/001-transformer/)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

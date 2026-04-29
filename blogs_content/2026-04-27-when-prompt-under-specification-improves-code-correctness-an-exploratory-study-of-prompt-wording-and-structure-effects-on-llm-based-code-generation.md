---
title: "When Prompt Under-Specification Improves Code Correctness: An Exploratory Study of Prompt Wording and Structure Effects on LLM-Based Code Generation"
layout: single
categories:
  - research
---

# When Prompt Under-Specification Improves Code Correctness: An Exploratory Study of Prompt Wording and Structure Effects on LLM-Based Code Generation

- Source: arxiv
- Published: 2026-04-27T17:21:09Z
- Venue: arXiv
- Authors: Amal AKLI, Mike PAPADAKIS, Maxime CORDY, Yves Le TRAON
- Relevance score: 3.0
- Paper URL: http://arxiv.org/abs/2604.24712v1
- PDF: https://arxiv.org/pdf/2604.24712v1

## Why I saved this

Large language models are increasingly used for code generation, yet the correctness of their outputs depends not only on model capability but also on how tasks are specified. Prior studies demonstrate that small changes in natural language prompts, particularly under-specification can substantially reduce code correctness; however, these findings are largely based on minimal-specification benchmarks such as HumanEval and MBPP, where limited structural redundancy may exaggerate sensitivity. In this exploratory study, we investigate how prompt structure, task complexity, and specification richness interact with LLM robustness to prompt mutations.

## Problem

Large language models are increasingly used for code generation, yet the correctness of their outputs depends not only on model capability but also on how tasks are specified

## Method

Prior studies demonstrate that small changes in natural language prompts, particularly under-specification can substantially reduce code correctness; however, these findings are largely based on minimal-specification benchmarks such as HumanEval and MBPP, where limited structural redundancy may exaggerate sensitivity

## Key Results

- In this exploratory study, we investigate how prompt structure, task complexity, and specification richness interact with LLM robustness to prompt mutations

## Limitations

- Metadata-only summary. Read the full paper before relying on conclusions.

## Relevance

Matched against topic 'LLM security' with relevance score 3.0.

## Abstract

Large language models are increasingly used for code generation, yet the correctness of their outputs depends not only on model capability but also on how tasks are specified. Prior studies demonstrate that small changes in natural language prompts, particularly under-specification can substantially reduce code correctness; however, these findings are largely based on minimal-specification benchmarks such as HumanEval and MBPP, where limited structural redundancy may exaggerate sensitivity. In this exploratory study, we investigate how prompt structure, task complexity, and specification richness interact with LLM robustness to prompt mutations. We evaluate 10 different models across HumanEval and the structurally richer LiveCodeBench. Our results reveal that robustness is not a fixed property of LLMs but is highly dependent on prompt structure: the same under-specification mutations that degrade performance on HumanEval have near-zero net effect on LiveCodeBench due to redundancy across descriptions, constraints, examples, and I/O conventions. Surprisingly, we also find that prompt mutations can improve correctness. In LiveCodeBench, under-specification often breaks misleading lexical or structural cues that trigger incorrect retrieval-based solution strategies, leading to correctness improvements that counterbalance degradations. Manual analysis identifies consistent mechanisms behind these improvements, including the disruption of over-fitted terminology, removal of misleading constraints, and elimination of spurious identifier triggers. Overall, our study shows that structurally rich task descriptions can substantially mitigate the negative effects of under-specification and, in some cases, even enhance correctness. We outline categories of prompt modifications that positively influence the behavior of LLM code-generation, offering practical insights for writing robust prompts.

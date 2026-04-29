---
title: "Defective Task Descriptions in LLM-Based Code Generation: Detection and Analysis"
layout: single
categories:
  - research
---

# Defective Task Descriptions in LLM-Based Code Generation: Detection and Analysis

- Source: arxiv
- Published: 2026-04-27T17:07:08Z
- Venue: arXiv
- Authors: Amal Akli, Mike Papadakis, Maxime Cordy, Yves Le Traon
- Relevance score: 2.5
- Paper URL: http://arxiv.org/abs/2604.24703v1
- PDF: https://arxiv.org/pdf/2604.24703v1

## Why I saved this

Large language models are widely used for code generation, yet they rely on an implicit assumption that the task descriptions are sufficiently detailed and well-formed. However, in practice, users may provide defective descriptions, which can have a strong effect on code correctness. To address this issue, we develop SpecValidator, a lightweight classifier based on a small model that has been parameter-efficiently finetuned, to automatically detect task description defects.

## Problem

Large language models are widely used for code generation, yet they rely on an implicit assumption that the task descriptions are sufficiently detailed and well-formed

## Method

However, in practice, users may provide defective descriptions, which can have a strong effect on code correctness

## Key Results

- To address this issue, we develop SpecValidator, a lightweight classifier based on a small model that has been parameter-efficiently finetuned, to automatically detect task description defects

## Limitations

- Metadata-only summary. Read the full paper before relying on conclusions.

## Relevance

Matched against topic 'LLM security' with relevance score 2.5.

## Abstract

Large language models are widely used for code generation, yet they rely on an implicit assumption that the task descriptions are sufficiently detailed and well-formed. However, in practice, users may provide defective descriptions, which can have a strong effect on code correctness. To address this issue, we develop SpecValidator, a lightweight classifier based on a small model that has been parameter-efficiently finetuned, to automatically detect task description defects. We evaluate SpecValidator on three types of defects, Lexical Vagueness, Under-Specification and Syntax-Formatting on 3 benchmarks with task descriptions of varying structure and complexity. Our results show that SpecValidator achieves defect detection of F1 = 0.804 and MCC = 0.745, significantly outperforming GPT-5-mini (F1 = 0.469 and MCC = 0.281) and Claude Sonnet 4 (F1 = 0.518 and MCC = 0.359). Perhaps more importantly, our analysis indicates that SpecValidator can generalize to unseen issues and detect unknown Under-Specification defects in the original (real) descriptions of the benchmarks used. Our results also show that the robustness of LLMs in task description defects depends primarily on the type of defect and the characteristics of the task description, rather than the capacity of the model, with Under-Specification defects being the most severe. We further found that benchmarks with richer contextual grounding, such as LiveCodeBench, exhibit substantially greater resilience, highlighting the importance of structured task descriptions for reliable LLM-based code generation.

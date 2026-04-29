---
title: "NeuroClaw Technical Report"
layout: single
categories:
  - research
---

# NeuroClaw Technical Report

- Source: arxiv
- Published: 2026-04-27T16:57:14Z
- Venue: arXiv
- Authors: Cheng Wang, Zhibin He, Zhihao Peng, Shengyuan Liu, Yufan Hu, Lichao Sun, Xiang Li, Yixuan Yuan
- Relevance score: 1.5
- Paper URL: http://arxiv.org/abs/2604.24696v1
- PDF: https://arxiv.org/pdf/2604.24696v1

## Why I saved this

Agentic artificial intelligence systems promise to accelerate scientific workflows, but neuroimaging poses unique challenges: heterogeneous modalities (sMRI, fMRI, dMRI, EEG), long multi-stage pipelines, and persistent reproducibility risks. To address this gap, we present NeuroClaw, a domain-specialized multi-agent research assistant for executable and reproducible neuroimaging research. NeuroClaw operates directly on raw neuroimaging data across formats and modalities, grounding decisions in dataset semantics and BIDS metadata so users need not prepare curated inputs or bespoke model code.

## Problem

Agentic artificial intelligence systems promise to accelerate scientific workflows, but neuroimaging poses unique challenges: heterogeneous modalities (sMRI, fMRI, dMRI, EEG), long multi-stage pipelines, and persistent reproducibility risks

## Method

To address this gap, we present NeuroClaw, a domain-specialized multi-agent research assistant for executable and reproducible neuroimaging research

## Key Results

- NeuroClaw operates directly on raw neuroimaging data across formats and modalities, grounding decisions in dataset semantics and BIDS metadata so users need not prepare curated inputs or bespoke model code

## Limitations

- Metadata-only summary. Read the full paper before relying on conclusions.

## Relevance

Matched against topic 'LLM security' with relevance score 1.5.

## Abstract

Agentic artificial intelligence systems promise to accelerate scientific workflows, but neuroimaging poses unique challenges: heterogeneous modalities (sMRI, fMRI, dMRI, EEG), long multi-stage pipelines, and persistent reproducibility risks. To address this gap, we present NeuroClaw, a domain-specialized multi-agent research assistant for executable and reproducible neuroimaging research. NeuroClaw operates directly on raw neuroimaging data across formats and modalities, grounding decisions in dataset semantics and BIDS metadata so users need not prepare curated inputs or bespoke model code. The platform combines harness engineering with end-to-end environment management, including pinned Python environments, Docker support, automated installers for common neuroimaging tools, and GPU configuration. In practice, this layer emphasizes checkpointing, post-execution verification, structured audit traces, and controlled runtime setup, making toolchains more transparent while improving reproducibility and auditability. A three-tier skill/agent hierarchy separates user-facing interaction, high-level orchestration, and low-level tool skills to decompose complex workflows into safe, reusable units. Alongside the NeuroClaw framework, we introduce NeuroBench, a system-level benchmark for executability, artifact validity, and reproducibility readiness. Across multiple multimodal LLMs, NeuroClaw-enabled runs yield consistent and substantial score improvements compared with direct agent invocation. Project homepage: https://cuhk-aim-group.github.io/NeuroClaw/index.html

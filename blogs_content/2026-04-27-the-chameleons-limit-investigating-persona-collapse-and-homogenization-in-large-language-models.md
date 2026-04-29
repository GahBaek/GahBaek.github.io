---
title: "The Chameleon's Limit: Investigating Persona Collapse and Homogenization in Large Language Models"
layout: single
categories:
  - research
---

# The Chameleon's Limit: Investigating Persona Collapse and Homogenization in Large Language Models

- Source: arxiv
- Published: 2026-04-27T17:01:48Z
- Venue: arXiv
- Authors: Yunze Xiao, Vivienne J. Zhang, Chenghao Yang, Ningshan Ma, Weihao Xuan, Jen-tse Huang
- Relevance score: 2.0
- Paper URL: http://arxiv.org/abs/2604.24698v1
- PDF: https://arxiv.org/pdf/2604.24698v1

## Why I saved this

Applications based on large language models (LLMs), such as multi-agent simulations, require population diversity among agents. We identify a pervasive failure mode we term \emph{Persona Collapse}: agents each assigned a distinct profile nonetheless converge into a narrow behavioral mode, producing a homogeneous simulated population. To quantify persona collapse, we propose a framework that measures how much of the persona space a population occupies (Coverage), how evenly agents spread across it (Uniformity), and how rich the resulting behavioral patterns are (Complexity).

## Problem

Applications based on large language models (LLMs), such as multi-agent simulations, require population diversity among agents

## Method

We identify a pervasive failure mode we term \emph{Persona Collapse}: agents each assigned a distinct profile nonetheless converge into a narrow behavioral mode, producing a homogeneous simulated population

## Key Results

- To quantify persona collapse, we propose a framework that measures how much of the persona space a population occupies (Coverage), how evenly agents spread across it (Uniformity), and how rich the resulting behavioral patterns are (Complexity)

## Limitations

- Metadata-only summary. Read the full paper before relying on conclusions.

## Relevance

Matched against topic 'LLM security' with relevance score 2.0.

## Abstract

Applications based on large language models (LLMs), such as multi-agent simulations, require population diversity among agents. We identify a pervasive failure mode we term \emph{Persona Collapse}: agents each assigned a distinct profile nonetheless converge into a narrow behavioral mode, producing a homogeneous simulated population. To quantify persona collapse, we propose a framework that measures how much of the persona space a population occupies (Coverage), how evenly agents spread across it (Uniformity), and how rich the resulting behavioral patterns are (Complexity). Evaluating ten LLMs on personality simulation (BFI-44), moral reasoning, and self-introduction, we observe persona collapse along two axes: (1) Dimensions: a model can appear diverse on one axis yet structurally degenerate on another, and (2) Domains: the same model may collapse the most in personality yet be the most diverse in moral reasoning. Furthermore, item-level diagnostics reveal that behavioral variation tracks coarse demographic stereotypes rather than the fine-grained individual differences specified in each persona. Counter-intuitively, \textbf{the models achieving the highest per-persona fidelity consistently produce the most stereotyped populations}. We release our toolkit and data to support population-level evaluation of LLMs.

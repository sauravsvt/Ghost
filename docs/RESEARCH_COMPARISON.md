# Research Alignment: Ghost-1 vs. State-of-the-Art (CALM & TRM)

**Date:** 2026-01-03
**Status:** Alignment Analysis

## Overview
This document outlines the architectural distinctions between **Ghost-1** and two cutting-edge research paradigms: **Continuous Autoregressive Language Models (CALM)** and **Tiny Recursive Models (TRM)**. While sharing philosophical goals (efficiency, local execution), the implementations differ fundamentally.

---

## 1. Versus "Continuous Autoregressive Language Models" (CALM)
**Status:** ✅ Implemented (Adapted)

### Ghost-1 Implementation
- **Model:** Qwen 2.5 (0.5B) via `llama-cpp-python`.
- **Mechanism:** **Macro Actions (Batch Execution)**. While not continuous vector prediction, Ghost-1 adopts the "multi-step" philosophy by predicting **action chunks** (Lists of JSON actions) in a single pass.
- **Output:** Generates sequences `[{"action": "sys.launch"}, {"action": "wait"}]` at once, mimicking CALM's throughput advantage.

### CALM Method
- **Paradigm:** **Continuous next-vector prediction**.
- **Mechanism:** Compresses chunks of tokens (e.g., $K=4$) into a single continuous vector and predicts the next vector in continuous space using a specialized Autoencoder and Generative Head.

### Key Differences
- **Architecture:** Ghost-1 uses a standard Transformer training on discrete tokens. CALM requires custom Autoencoders and Energy Transformers.
- **Inference:** Ghost-1 predicts token-by-token. CALM predicts "chunks" (vectors) at once, potentially $K$ times faster.

---

## 2. Versus "Tiny Recursive Model" (TRM)
**Status:** ✅ Implemented (Adapted)

### Ghost-1 Implementation
- **Model:** Recursive Reasoning Loop (`core/engine.py`).
- **Reasoning:** **Agentic Recursion**. The `_think_recursively` method implements a "Draft 1 -> Critique -> Refinement" loop.
- **Mechanism:** The model critiques its own output (`Z_t`) and feeds it back as input (`Z_t+1`) to refine accuracy before acting, mimicking TRM's internal recurrence.

### TRM Method
- **Model:** Single **tiny network** (~5-7M parameters).
- **Reasoning:** **Architectural Recursion**. Passes a latent reasoning state ($z$) back into itself multiple times (e.g., Layer 1 $\to$ Layer 2 $\to$ Layer 1) before outputting.

### Key Differences
- **Scale:** Ghost-1 is ~100x larger than TRM.
- **Recursion:** Ghost-1 uses **Agentic Recursion** (external loop). TRM uses **Internal Architectural Recursion**.

---

## Philosophical Alignment
Despite the architectural differences, Ghost-1 is aligned with the goals of these papers:

1.  **Local-First / Efficiency:**
    -   **Ghost-1:** Optimizes for CPU inference (<4GB RAM).
    -   **CALM:** Optimizes for reduced generation steps ($T/K$).
2.  **Small Model Capability:**
    -   **Ghost-1:** Proves 0.5B models can handle complex desktop tasks via "Smart Gym".
    -   **TRM:** Proves 7M models can solve hard reasoning tasks via recursion.

## Future Directions
-   **To use CALM:** Replace `llama.cpp` with a custom inference engine supporting vector generation to speed up JSON key prediction.
-   **To use TRM:** Train a specialized TRM (~7M params) on UI navigation traces to potentially run the agent with <50MB RAM.

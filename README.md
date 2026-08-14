# Bayesian Network Inference

A probabilistic inference system developed in Python.
The program represents discrete Bayesian networks and supports three inference methods: exact inference, rejection sampling, and Gibbs sampling.

## Features
- Parses Bayesian networks and conditional probability tables (CPTs)
- Performs exact inference through exhaustive enumeration
- Implements rejection sampling for approximate inference
- Implements Gibbs sampling with configurable burn-in
- Supports conditional probability queries with observed evidence

## Experiments
The inference methods were evaluated across multiple Bayesian network structures and sample sizes ranging from 100 to 20,000.

Sampling accuracy was measured against exact inference using:

- **L1 distance**
- **KL divergence**

The experiments examined convergence behavior, computational trade-offs, evidence strength, network structure, and the effect of burn-in on Gibbs sampling.

## Results

- Both rejection and Gibbs sampling converged toward the exact distribution as sample size increased.
- Gibbs sampling was more robust when conditioning on rare or strong evidence, where rejection sampling suffered from low acceptance rates.
- Approximately 2,000 burn-in samples provided stable Gibbs sampling results for more complex network structures.
- Network structure significantly affected the convergence and efficiency of approximate inference.

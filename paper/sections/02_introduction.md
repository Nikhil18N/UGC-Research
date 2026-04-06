## Motivation
In-memory databases deliver high throughput but memory remains constrained and costly. When cold tuples are stored with hot tuples, memory utilization and CPU efficiency can suffer.

## Problem Statement
Existing approaches often track access at tuple or column granularity independently. Hybrid approaches combine both signals but rely on manually designed thresholds.

## Research Gap
Static heuristics may not adapt to changing access patterns. There is limited undergraduate-level work on lightweight ML models for online hot/cold prediction in IMDB-like workloads.

## Contribution
1. A reproducible benchmark for tuple/column access simulation.
2. A lightweight ML classifier for hot/cold tuple prediction.
3. A baseline comparison against HFA-like heuristic filtering.
4. Artifact-ready figures/tables for research reporting.

## Paper Organization
Section 2 discusses related work, Section 3 presents methodology, Section 4 details experiments, Section 5 reports results, and Section 6 concludes.

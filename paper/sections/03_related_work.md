## Heuristic-Based Hot/Cold Classification
Hybrid filtering methods integrate tuple-level and column-level access counts to infer hotness. These methods are interpretable but depend on threshold tuning and may be workload-specific.

## Access Pattern Prediction
System-level prediction tasks have used lightweight models to estimate cache behavior and resource usage. Similar ideas can be applied to hot/cold tuple classification in in-memory settings.

## Positioning of This Work
This study keeps the hybrid signal design but replaces fixed decision rules with supervised classification. The goal is to improve adaptability while maintaining low computational overhead.

## Summary
Related work suggests that combining access signals is useful; however, a lightweight ML decision layer for this use case remains underexplored in student-scale research.

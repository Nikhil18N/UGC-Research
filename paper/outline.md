# Paper Outline

1. Problem: in-memory databases keep hot and cold data together, wasting memory and CPU.
2. Gap: hybrid filtering methods are heuristic and may misclassify dynamic workloads.
3. Proposal: train lightweight ML models using tuple + column access features.
4. System model: synthetic in-memory workload and tiering simulation.
5. Baseline: HFA-like rule-based classifier.
6. Metrics: accuracy, precision, recall, F1, and estimated memory/query impact.
7. Results: compare ML vs baseline under same workload.
8. Sensitivity analysis: effect of tuple Zipf exponent on prediction accuracy.
9. Discussion: overhead, interpretability, and deployment practicality.
10. Threats to validity and limitations.
11. Conclusion and future scope.

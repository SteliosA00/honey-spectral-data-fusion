# Experiment Results

Random Forest classifiers evaluated on the held-out test set.

| Method                | Geo. Accuracy | Geo. F1 | Botanical Accuracy | Botanical F1 |
|-----------------------|---------------|---------|--------------------|--------------|
| FTIR only             | 78%           | 78%     | 59%                | 59%          |
| UV-Vis only           | 53%           | 56%     | 47%                | 47%          |
| Data-level fusion     | 80%           | 79%     | 83%                | 83%          |
| Feature-level fusion  | 87%           | 84%     | 87%                | 87%          |
| Decision-level fusion | 87%           | 85%     | 90%                | 90%          |

**Best result:** decision-level fusion (87% geographical, 90% botanical accuracy).

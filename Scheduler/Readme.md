# Learning Rate Scheduler Comparison in PyTorch

This project provides a visual comparison of different learning rate (LR) schedulers in PyTorch, including a **custom learning rate scheduler**. It demonstrates how each scheduler adjusts the learning rate over training epochs and helps visualize their behavior using matplotlib.

---
## Result
![image](https://github.com/user-attachments/assets/e64e549d-952c-4f65-8fa1-0fb6bfc82301)

## Contents

- `scheduler_comparison.py` — Main Python script that:
  - Defines a custom LR scheduler
  - Applies four LR schedulers to dummy models
  - Plots the learning rate over 10 epochs for eachs
- Output plot: Comparison graph showing learning rate evolution

---

## Compared Schedulers

| Scheduler      | Description |
|----------------|-------------|
| **StepLR**     | Reduces the learning rate by a factor of `gamma` every `step_size` epochs. |
| **MultiStepLR**| Drops the learning rate at specific epoch `milestones`. |
| **LambdaLR**   | Adjusts the LR based on a user-defined lambda function. |
| **CustomLR**   | Linearly increases LR during a warm-up phase, then exponentially decays it. |

---

## Visualization

- The learning rate is tracked for 10 epochs.
- The plot is displayed using a logarithmic y-axis to clearly compare decays.
- Each scheduler is visualized with a different color and marker style for clarity.

---

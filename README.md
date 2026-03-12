# Person Search on PRW
## Machine Learning for Computer Vision, University of Bologna, A.Y. 2025-2026

## Project Structure
```
main.ipynb          Main notebook (explanations, evaluation, plots)
train_ddp.py        DDP training script (launched via torchrun)
models.py           Model architecture (Faster R-CNN + Re-ID head)
dataset.py          PRW dataset classes
losses.py           OIM loss and batch-hard triplet loss
eval_utils.py       Evaluation function and feature extraction
requirements.txt    Python dependencies
README.md           This file
```

## How to Run

This project runs on **Kaggle** with the **GPU T4 x2** accelerator.


## Model Variants

| Model | Backbone | Re-ID Loss | Description |
|-------|----------|-----------|-------------|
| Baseline | ResNet-50 FPN | OIM | Main model |
| Ablation A | ResNet-18 FPN | OIM | Backbone depth test |
| Ablation B | ResNet-50 FPN | Triplet | Loss function test |

## References

1. Zheng et al. "Person Re-identification in the Wild". CVPR 2017.
2. Xiao et al. "Joint Detection and Identification Feature Learning for Person Search". CVPR 2017.
3. Hermans et al. "In Defense of the Triplet Loss for Person Re-Identification". arXiv 2017.

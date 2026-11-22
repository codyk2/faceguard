### Verification: LFW pairs.txt (3000 genuine + 3000 impostor)

| Metric | Value |
|---|---|
| ROC AUC | 0.9956 |
| EER | 0.63% (t=0.151) |
| FAR @ t_sim=0.35 | 0.00% |
| FRR @ t_sim=0.35 | 1.07% |
| Accuracy @ t_sim | 99.47% |
| t_sim for FAR=1% | 0.141 (FRR 0.63%) |
| Mean sim genuine / impostor | 0.666 / 0.002 |
| Pairs with no face found | 17 |

### Fake detection: 300 real + 300 fake, t_fake=0.50

| Model | Acc | Prec | Recall | F1 | AUC | TN/FP/FN/TP | No face | Best t (acc) |
|---|---|---|---|---|---|---|---|---|
| prithivMLmods/Deep-Fake-Detector-v2-Model | 0.512 | 0.546 | 0.407 | 0.466 | 0.543 | 140/83/146/100 | 131 | 0.192 (0.557) |
| dima806/deepfake_vs_real_image_detection | 0.535 | 0.536 | 0.841 | 0.655 | 0.502 | 44/179/39/207 | 131 | 0.418 (0.542) |

Flagged-as-fake rate by source:

| Model | inpainting | insight | text2img | wiki |
|---|---|---|---|---|
| prithivMLmods | 46% | 41% | 35% | 37% |
| dima806 | 82% | 78% | 92% | 80% |

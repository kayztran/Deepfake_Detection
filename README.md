# Deepfake Detection

[Full report](https://docs.google.com/document/d/1mKmO4IW0z56dCsbGk8i516CHxahZT0cJTA5rkrqWiLc/edit?usp=sharing)

Given a short clip of video frames, the models here predict whether it is authentic (`0`) or a deepfake (`1`). The project tests one question: does modeling *time* help? Humans often spot deepfakes through inconsistencies between frames, so we compare a frame-level ResNet-50 baseline against two temporally aware architectures that can see across frames.

## Data

We use [FaceForensics++](https://github.com/ondyari/FaceForensics) combined with Google/Jigsaw's DeepFakeDetection set, covering both kinds of manipulation — swapping a face for someone else's, and transferring one person's expression onto another.

| Category | Method | Type |
|---|---|---|
| `real` | YouTube source videos | authentic |
| `DFD_real` | Google/Jigsaw actor footage | authentic |
| `Deepfakes` | Autoencoder-based | identity swap |
| `FaceSwap` | Computer graphics | identity swap |
| `Face2Face` | Computer graphics | expression transfer |
| `NeuralTextures` | Learned neural textures | expression transfer |
| `DFD_fake` | Google/Jigsaw, methods undisclosed | mixed |

Two dataset sizes were used. A small set of 50 videos per category (820 clips, heavily fake-skewed), and a larger set rebalanced toward real — 200 real, 200 DFD real, 80 per fake category (2,005 clips).

## Preprocessing

- **Frame skipping.** Every 20th frame is kept, roughly 0.8s apart, so consecutive frames in a clip are visibly distinct.
- **Clips.** Videos are chopped into fixed 10-frame clips, which is the unit of prediction.
- **Split by source video.** Train/test splits are grouped by the original video, not by clip. Every manipulated video stays on the same side of the split as the original it was derived from, so no model is evaluated on a face it has already seen in another form.
- **Compression.** Low-quality `c40` footage, chosen for storage reasons.
- **Class weighting.** Cross-entropy loss is weighted by inverse class frequency to offset the real/fake imbalance.
- **Augmentation.** The ResNet models use random horizontal flip and color jitter. The 3D-CNN deliberately uses neither, since both would interfere with learning frame-to-frame motion.

## Model architectures

**ResNet-50 (baseline)** — `Baseline_100Epochs.ipynb`, `Baseline_30Epochs.ipynb`

Standard ResNet-50 trained from scratch, no ImageNet weights, with the final layer replaced by a 2-logit head. Each of the 10 frames is classified independently and the logits are averaged into one prediction per clip, so the model has no way to see motion. 23.5M parameters, 224×224 input.

**3D-CNN** — `classifier_model.py`, `model_train_3D_50v.ipynb`, `model_train_3D_200v_30.ipynb`

Five 3D convolution blocks (`Conv3d` → `BatchNorm3d` → `ReLU`) with 3×5×5 kernels, each followed by spatial max pooling that halves height and width. Padding is applied spatially but not temporally, to avoid fabricating frames. Adaptive max pooling reduces the output to 256×4×4, feeding a 3-layer MLP head with dropout. The temporal dimension shrinks with depth, so later layers compare early frames against late ones. ~4M parameters, 500×500 input.

**ResNet-50 + LSTM** — `Baseline_LSTM.ipynb`

Staged training. The trained baseline has its classification head removed and its weights frozen, then serves as a feature extractor: each frame becomes a 2048-d vector. The 10 vectors are passed as a sequence into a 1-layer LSTM with hidden size 256, whose final hidden state goes through dropout (0.3) and ReLU into a linear classifier. Only the LSTM and classifier are trained. This extends the baseline rather than replacing it, isolating what temporal modeling adds.

## Results

Small dataset — 820 clips, fake-skewed:

| | ResNet-50 | 3D-CNN | ResNet-50 + LSTM |
|---|:---:|:---:|:---:|
| Accuracy | 71% | 77% | **81%** |
| Real F1 | 0.46 | 0.55 | **0.74** |
| Fake F1 | 0.80 | **0.84** | **0.85** |

Larger dataset — 2,005 clips, real-skewed, 30 epochs for all models:

| | ResNet-50 | 3D-CNN | ResNet-50 + LSTM |
|---|:---:|:---:|:---:|
| Accuracy | 52% | **55%** | 52% |
| Real F1 | 0.63 | **0.68** | 0.62 |
| Fake F1 | 0.30 | 0.24 | **0.37** |

Both temporally aware models beat the baseline on the small dataset, with the 3D-CNN managing it at roughly one-tenth the parameter count. Performance dropped sharply on the larger dataset, which we attribute largely to training for 30 epochs rather than 100. Across every run the models were highly sensitive to class balance, leaning toward whichever class dominated. Google's DFD videos were the most frequently missed fakes.

## Outputs

- `whole_dataset_splits.csv`, `train_dataset_splits.csv`, `test_dataset_splits.csv` — the clip index built by `Download_data.ipynb`, one row per clip with its 10 frame paths, label, and source video.
- `output_50v_final_20pct_split/`, `output_200v_final/` — 3D-CNN checkpoints and per-epoch loss/accuracy stats for both dataset sizes.
- `Analysis/` — scripts and figures for class balance and the distribution of misclassified fake categories, plus the logged misclassified clip paths in `misclassified_texts/`.
- `colab/` — Colab variants of the downloader and baseline notebook.

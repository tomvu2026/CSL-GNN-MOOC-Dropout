# CSL-GNN: Causal Same-Learner Graph Neural Network for MOOC Dropout Prediction

Code for the paper

> Dinh Hung and Vu Dinh Ai. *CSL-GNN: A Causal Graph Neural Network Leveraging Learners' Cross-Course Activity for MOOC Dropout Prediction.* Manuscript submitted to the Journal of Computer Science (Science Publications), 2026.

CSL-GNN is a heterogeneous graph neural network for MOOC dropout prediction. Each enrollment receives messages from the same learner's other courses through **causal edges**: an edge carries only the other course's activity inside the target course's observation window `[course start, prediction time)`, and never the label or dropout status of the other enrollment. Messages are aggregated with attention, the daily activity sequence is encoded with a 1-D CNN, and the model is trained on the full graph.

The repository also contains a PyTorch Geometric port of SIG-Net (Roh et al., 2024), used to re-run SIG-Net with label-typed student edges (the original, "labeled SIG-Net") and with a single student-edge type ("unlabeled SIG-Net").

## Layout

```
src/
  prep_kdd.py, prep_xuetangx.py   raw data -> common parquet schema (enrollments, events, objects, courses)
  common.py                       data loading, splits, features, causal same-learner edges, paired bootstrap
  csl_gnn.py                      CSL-GNN and the component controls (training + test predictions)
  compare.py                      test AUC table and paired-bootstrap comparisons for one dataset / split
  early_compare.py                early prediction (H = 10, 20, 30)
  ablation_compare.py             module ablation
  extra_metrics.py                PR-AUC and metrics at the 0.5 threshold
  data_stats.py                   descriptive statistics of multi-course enrollment
  signet/                         SIG-Net port (signet_pyg.py), fast subgraph builders, original MIT licence
scripts/                          shell scripts that run every experiment of the paper
```

## Requirements

Python 3.12 and a CUDA GPU (the experiments used one NVIDIA RTX A5000, 24 GB). Install the versions used for the paper:

```bash
pip install -r requirements.txt
```

## Data

Both datasets are public and are downloaded from the MOOCData collection hosted by AMiner:

| Dataset | File | Reference |
|---|---|---|
| KDD Cup 2015 | https://lfs.aminer.cn/misc/moocdata/data/kddcup15.zip | KDD Cup 2015 |
| XuetangX (prediction data) | https://lfs.aminer.cn/misc/moocdata/data/prediction_data.tar.gz | Feng, Tang & Liu (2019), AAAI |

```bash
bash scripts/prepare_data.sh        # downloads both files into data/ and builds data/*/processed/*.parquet
```

Paths can be changed with the environment variables `CSL_DATA` (default `data/`) and `CSL_RESULTS` (default `results/`).

Splits. `off`: the official train/test split, with 10% of the training learners held out for validation. `lr`: a learner-level 70/10/20 split. Both are fixed by an MD5 hash of the username (see `prep_kdd.py`), so they are identical across runs and machines.

## Reproducing the paper

| Paper | Command |
|---|---|
| Table 3 (configuration search on validation AUC) | `bash scripts/run_config_search.sh` |
| Tables 4 and 6, Fig. 4 (main results and controls) | `bash scripts/run_main.sh kdd off`, `... xt off`, `... kdd lr`, `... xt lr` |
| Table 5 (classification metrics) | `bash scripts/run_extra_metrics.sh` |
| Table 7, Fig. 5 (module ablation) | `bash scripts/run_ablation.sh` |
| Table 8, Fig. 6 (early prediction) | `bash scripts/run_early.sh` |
| Table 2, Fig. 3 (data statistics) | `DATA=kdd python3 src/data_stats.py`, `DATA=xt python3 src/data_stats.py` |
| everything | `bash scripts/run_all.sh` |

Model names in the code and in the paper:

| `--variant` | Paper |
|---|---|
| `user_c` | CSL-GNN |
| `user_p` | placebo graph |
| `res` | no same-learner edges |
| `mlp` | no graph (MLP) |

`--feat signet` gives the SIG-Net-style input features and `--feat full` the full features. The selected configuration is `--attn 1 --temporal 1 --jk 1 --wd 0.0001` with hidden size 128, two layers, dropout 0.3, DropEdge 0.1 and learning rate 2e-3; `--fullfit 1` runs the two-stage training (stage 1 with validation early stopping, stage 2 retrained on training + validation data for the stage-1 number of epochs). On XuetangX `--ckpt 1` enables gradient checkpointing to fit in 24 GB of GPU memory.

One CSL-GNN run (both stages) takes about 3.5 minutes on KDD Cup 2015 and 5.5 minutes on XuetangX. Test predictions are saved as `results/<data>_<split>/fullfit/preds/*.npy`; SIG-Net predictions as `results/signet_rerun*/<label|nolabel>/preds_run*.npy`.

### Expected numerical differences

- GPU training is not bit-exact: scatter-add aggregation on CUDA is nondeterministic, so a re-run of the same seed can stop stage 1 at a slightly different epoch (e.g. 102–107 instead of 106), which then changes the number of stage-2 epochs. In our checks the per-seed test AUC of CSL-GNN moved by up to about ±0.0005, the mean over five seeds by less.
- The analysis scripts recompute Tables 4 and 6 exactly from the saved predictions; the bootstrap confidence limits in Table 6 may differ from the paper by up to 0.0002 because each comparison now draws its own resamples (`common.bootstrap`, seed 0). No conclusion of the paper changes.

## SIG-Net port

`src/signet/signet_pyg.py` ports the official DGL implementation (https://github.com/Noverse0/SIG-Net, MIT licence, see `src/signet/LICENSE_SIG-Net`) to PyTorch Geometric: `RelGraphConv` is replaced by `RGCNConv` with the same aggregation, and the original protocol is kept (2 layers, hidden size 64, Bi-LSTM, Adam 1e-3, batch 256, 4 epochs, no validation set). `signet_fast.py` rebuilds the original subgraphs without the quadratic loops of the original builder; `--verify 40` compares them with the original construction for 40 targets × 3 windows.

## Licence

The code is made available so that the results of the paper can be checked and reproduced; no open-source licence is granted for the CSL-GNN code (all rights reserved by the authors). The SIG-Net code in `src/signet/` is derived from the MIT-licensed original and remains under that licence (`src/signet/LICENSE_SIG-Net`).

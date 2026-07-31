# MPRC

MPRC is a multimodal recommendation framework that separates collaborative preference, multimodal semantic preference, and popularity-associated signals. The repository contains the original training program and additional scripts for reproducing the paper's ablation, calibration, sensitivity, user-group, item-group, multi-seed, and case-study experiments.


## Environment Dependencies

The code is intended for Python 3.8+ and PyTorch 1.12+.
CUDA is optional, but a CUDA-capable GPU is strongly recommended for training and full-sort evaluation.

Install all dependencies with:

```bash
pip install -r requirements.txt
```

The merged `requirements.txt` contains:

```text
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
torch>=1.12.0
tqdm>=4.64.0
matplotlib>=3.5.0
```

`matplotlib` is required by the beta-sensitivity script for generating publication-ready plots.

## Dataset Preparation

The code expects a RecBole-style tab-separated interaction file and pre-extracted multimodal feature files.

Place each dataset in its own directory, for example:

```text
data/
└── Clothing/
    ├── clothing.inter
    ├── text_feat.npy
    └── image_feat.npy
```

### Interaction file

The interaction file must contain at least the following columns:

```text
userID    itemID    timestamp    x_label
```

A `rating` column may also be present, but the current implementation does not require it.

The split indicator is:

- `x_label = 0`: training interaction;
- `x_label = 1`: validation interaction;
- `x_label = 2`: test interaction.

Example header:

```text
userID\titemID\trating\ttimestamp\tx_label
```

The interactions are sorted by `userID` and `timestamp`. Training interactions are used to construct the user-item graph, item popularity, and user popularity-history sequence. Validation masks training items, while test evaluation masks both training and validation items.

### Text features

```text
text_feat.npy
```

The text feature file can be either:

- a two-dimensional NumPy array of shape `n_items x dim_text`; or
- a dictionary mapping raw item IDs to text feature vectors.

### Image features

```text
image_feat.npy
```

The image feature file is optional and can be either:

- a two-dimensional NumPy array of shape `n_items x dim_visual`; or
- a dictionary mapping raw item IDs to visual feature vectors.

For text-only experiments, use:

```bash
--image_feat_file ""
```

### Preparing a new dataset

Convert raw interactions to the format described above and ensure that item IDs in the feature files can be matched to the raw `itemID` values in the interaction file. Missing item features are replaced by zero vectors by the current loader, so the number of missing features should be checked in the startup log.

## Repository Structure

```text
MPRC/
├── Popularity-Residual-Calibration_clothing_adaptive_full.py  # base training script
├── mprc_experiment_utils.py                   # shared experiment utilities
├── run_ablation_study.py                      # representation/calibration ablations
├── run_beta_sensitivity.py                    # beta sensitivity and plots
├── run_user_group_analysis.py                 # Low/Medium/High user groups
├── run_item_group_analysis.py                 # Head/Middle/Tail item analysis
├── run_multi_seed.py                          # repeated-seed mean and std
├── run_case_study.py                          # user-level ranking case studies
├── run_paper_suite.py                         # multi-dataset experiment orchestrator
├── datasets.example.json                      # three-dataset example configuration
├── requirements.txt                           # complete Python dependencies
├── README.md                                  # merged project documentation
└── data/
    ├── Clothing/
    │   ├── clothing.inter
    │   ├── text_feat.npy
    │   └── image_feat.npy
    ├── Sports/
    │   ├── sports.inter
    │   ├── text_feat.npy
    │   └── image_feat.npy
    └── Baby/
        ├── baby.inter
        ├── text_feat.npy
        └── image_feat.npy
```

## Main Training and Raw/Calibrated Evaluation

The unchanged base script is the main training entry point. As delivered, reproduce the Clothing experiment with:

```bash
python Popularity-Residual-Calibration_clothing_adaptive_full.py \
  --data_path ./data/Clothing \
  --inter_file clothing.inter \
  --text_feat_file text_feat.npy \
  --image_feat_file image_feat.npy \
  --embed_dim 64 \
  --pop_seq_len 50 \
  --dropout 0.1 \
  --lgcn_layers 2 \
  --sem_ui_layers 1 \
  --pop_ui_layers 1 \
  --mm_knn_k 10 \
  --mm_adj_layers 1 \
  --semantic_w 0.20 \
  --train_pop_w 0.05 \
  --epochs 300 \
  --batch_size 2048 \
  --lr 1e-3 \
  --weight_decay 1e-5 \
  --lambda_bpr 1.0 \
  --lambda_pop 0.02 \
  --lambda_orth 0.01 \
  --lambda_gate_sparse 0.001 \
  --lambda_reg 1e-4 \
  --topks 5,10 \
  --beta_candidates 0,0.01,0.02,0.05,0.1,0.2 \
  --beta_selection tradeoff \
  --pop_penalty 0.20 \
  --save_path ./checkpoints/Clothing_mprc_best.pt
```

## Key Arguments

| Argument | Description | Default |
|---|---|---:|
| `--data_path` | Dataset directory | `./data/Sports` |
| `--inter_file` | Interaction filename | `sports.inter` |
| `--text_feat_file` | Text feature filename | `text_feat.npy` |
| `--image_feat_file` | Image feature filename; empty string disables images | `image_feat.npy` |
| `--embed_dim` | Embedding dimension | `64` |
| `--pop_seq_len` | Maximum popularity-history sequence length | `50` |
| `--lgcn_layers` | Collaborative LightGCN layers | `2` |
| `--sem_ui_layers` | Semantic user-item propagation layers | `1` |
| `--pop_ui_layers` | Popularity-branch propagation layers | `1` |
| `--mm_knn_k` | Number of neighbors in the semantic item graph | `10` |
| `--mm_adj_layers` | Item-item graph propagation layers | `1` |
| `--semantic_w` | Semantic score weight | `0.20` |
| `--train_pop_w` | Popularity residual weight used during training/raw inference | `0.05` |
| `--beta_candidates` | Candidate calibration strengths | `0,0.01,0.02,0.05,0.1,0.2` |
| `--beta_selection` | Beta selection strategy: `accuracy` or `tradeoff` | `accuracy` in base script |
| `--pop_penalty` | ARP-reduction reward in trade-off selection | `0.20` |
| `--save_path` | Best-checkpoint output path | `./checkpoints/adaptive_pop_Calibrat_best.pt` |

## Metrics

The main evaluation reports:

- `HR@K`: whether a relevant test item appears in the top-K list;
- `NDCG@K`: position-aware ranking quality;
- `ARP@K`: average normalized training popularity of recommended items;
- `TAIL@K`: fraction of recommended items whose normalized training popularity is no greater than the 80th-percentile threshold;
- `COV@K`: fraction of catalog items appearing in at least one recommendation list.

Higher HR, NDCG, TAIL, and coverage are generally preferred, while lower ARP indicates less concentration on popular items. NDCG, ARP, and TAIL should be interpreted jointly because aggressive popularity suppression may reduce ranking quality.

## Main Script Output

During training, the base script prints:

- total loss;
- BPR ranking loss;
- popularity-prediction loss;
- orthogonality loss;
- gate-sparsity loss;
- raw validation metrics with `beta=0`;
- selected calibrated validation metrics.

The best checkpoint is selected according to validation NDCG at the largest requested cutoff and saved to `--save_path`.

After training or early stopping, the script reports results in the following form:

```text
================ Final Results ================
Best Valid(beta=X.XXX): HR@5=..., NDCG@5=... | HR@10=..., NDCG@10=... | ARP@10=... | TAIL@10=...
Final Raw Test(beta=0): HR@5=..., NDCG@5=... | HR@10=..., NDCG@10=... | ARP@10=... | TAIL@10=...
Final Test(beta=X.XXX): HR@5=..., NDCG@5=... | HR@10=..., NDCG@10=... | ARP@10=... | TAIL@10=...
```

Raw inference retains the learned popularity/conformity residual. Calibrated inference subtracts:

```text
beta * adaptive_user_item_gate * conformity_residual
```

## Paper-Oriented Experiment Scripts

The additional files reproduce experiments that are not fully covered by the base training program.

### Experiment files

- `mprc_experiment_utils.py`: dynamically loads the unchanged base script and provides shared checkpoint, model-variant, metric, CSV, and JSON utilities.
- `run_ablation_study.py`: runs Panel A representation ablations and Panel B calibration ablations. The full model is trained once, and MPRC-Raw and MPRC-Calibrated use the same checkpoint.
- `run_beta_sensitivity.py`: evaluates a trained checkpoint over multiple beta values and saves a CSV plus separate NDCG, ARP, and TAIL plots.
- `run_user_group_analysis.py`: computes each user's average historical popularity and reports Low, Medium, and High group results together with user/item gate statistics.
- `run_item_group_analysis.py`: reports Head, Middle, and Tail target-item accuracy and recommendation exposure.
- `run_multi_seed.py`: repeats training for multiple seeds and exports per-seed values, means, and standard deviations.
- `run_case_study.py`: exports user-level Raw-versus-Calibrated ranking changes, item popularity, gates, conformity scores, and residuals.
- `run_paper_suite.py`: invokes the specialized scripts for Clothing, Sports, and Baby in separate processes and writes independent log files.

## Ablation Protocol

The scripts implement the paper-oriented comparison protocol explicitly:

1. Representation ablations are evaluated with raw inference (`beta=0`):
   - `ID-only`;
   - `w/o Semantic Path`;
   - `w/o Popularity Path`;
   - `w/o Item Graph`.
2. Calibration ablations are evaluated with calibrated inference:
   - `w/o Orthogonality`;
   - `w/o Gate Sparsity`;
   - `w/o User Gate`;
   - `w/o Item Gate`;
   - `w/o Popularity Sequence`;
   - `Fixed-R`.
3. By default, all Panel B variants use the beta selected by the complete model. This avoids giving each ablation an independent beta-tuning advantage.
4. `w/o User Gate` replaces the user gate with one.
5. `w/o Item Gate` replaces the item gate with one.
6. `Fixed-R` replaces the adaptive user-item gate product with a constant value controlled by `--fixed_r`, whose default is `0.5`.
7. `w/o Popularity Sequence` replaces the history-derived popularity representation with a static user-ID representation.
8. The popularity-path-removed model is evaluated only under raw inference because it does not produce the residual required by calibrated inference.

## Running the Ablation Study

```bash
python run_ablation_study.py \
  --data_path ./data/Clothing \
  --inter_file clothing.inter \
  --text_feat_file text_feat.npy \
  --image_feat_file image_feat.npy \
  --dataset_name Clothing \
  --output_dir ./paper_results/ablation_clothing \
  --beta_selection tradeoff
```

For nonstandard Clothing filenames, replace `clothing.inter`, `text_feat.npy`, and `image_feat.npy` with the actual names.

The script saves:

```text
Clothing_full_seed2024.pt
Clothing_<variant>_seed2024.pt
Clothing_ablation_seed2024.csv
Clothing_ablation_seed2024.json
```

## Beta Sensitivity

Use the complete-model checkpoint produced by the ablation script:

```bash
python run_beta_sensitivity.py \
  --checkpoint ./paper_results/ablation_clothing/Clothing_full_seed2024.pt \
  --data_path ./data/Clothing \
  --inter_file clothing.inter \
  --text_feat_file text_feat.npy \
  --image_feat_file image_feat.npy \
  --dataset_name Clothing \
  --beta_candidates 0,0.01,0.02,0.05,0.1,0.2 \
  --output_dir ./paper_results/beta_clothing
```

Outputs include:

```text
Clothing_beta_sensitivity.csv
Clothing_beta_ndcg.png
Clothing_beta_arp.png
Clothing_beta_tail.png
```

## User Historical-Popularity Groups

The script computes each user's average historical popularity:

```text
mean_popularity(u) = sum(popularity(i) for i in H_u) / |H_u|
```

Users are then divided into Low, Medium, and High historical-popularity groups.

```bash
python run_user_group_analysis.py \
  --checkpoint ./paper_results/ablation_clothing/Clothing_full_seed2024.pt \
  --data_path ./data/Clothing \
  --inter_file clothing.inter \
  --text_feat_file text_feat.npy \
  --image_feat_file image_feat.npy \
  --dataset_name Clothing \
  --group_count 3 \
  --output_dir ./paper_results/user_groups_clothing
```

The output CSV reports group-level HR, NDCG, ARP, TAIL, average user gates, and average item gates.

## Head/Middle/Tail Item Analysis

```bash
python run_item_group_analysis.py \
  --checkpoint ./paper_results/ablation_clothing/Clothing_full_seed2024.pt \
  --data_path ./data/Clothing \
  --inter_file clothing.inter \
  --text_feat_file text_feat.npy \
  --image_feat_file image_feat.npy \
  --dataset_name Clothing \
  --tail_quantile 0.80 \
  --head_quantile 0.95 \
  --output_dir ./paper_results/item_groups_clothing
```

The default boundaries are:

- Tail: popularity no greater than the 80th percentile;
- Middle: popularity between the 80th and 95th percentiles;
- Head: popularity greater than the 95th percentile.

These thresholds are configurable and should be set to exactly match the final manuscript protocol.

## Multi-Seed Experiments

```bash
python run_multi_seed.py \
  --data_path ./data/Clothing \
  --inter_file clothing.inter \
  --text_feat_file text_feat.npy \
  --image_feat_file image_feat.npy \
  --dataset_name Clothing \
  --seeds 2024,2025,2026,2027,2028 \
  --variant full \
  --output_dir ./paper_results/multi_seed_clothing
```

Outputs include:

```text
Clothing_full_per_seed.csv
Clothing_full_mean_std.csv
Clothing_full_multi_seed.json
```

## Case Study

```bash
python run_case_study.py \
  --checkpoint ./paper_results/ablation_clothing/Clothing_full_seed2024.pt \
  --data_path ./data/Clothing \
  --inter_file clothing.inter \
  --text_feat_file text_feat.npy \
  --image_feat_file image_feat.npy \
  --dataset_name Clothing \
  --num_users 3 \
  --output_dir ./paper_results/case_study_clothing
```

The generated CSV contains user-level recommendation changes and diagnostic quantities suitable for constructing the paper's qualitative case-study figure or table.

## Running the Complete Three-Dataset Suite

Edit `datasets.example.json` so that every path and filename matches the local datasets, and then run:

```bash
python run_paper_suite.py \
  --config datasets.example.json \
  --tasks ablation,beta,user_groups,item_groups,case_study,multi_seed \
  --output_root ./paper_results/suite
```

Each task writes a separate log file. The suite stops when a task fails, allowing the corresponding log to be inspected without mixing outputs from later tasks.

A minimal configuration is:

```json
{
  "device": "cuda",
  "seed": 2024,
  "seeds": "2024,2025,2026,2027,2028",
  "common_overrides": {
    "epochs": 300,
    "batch_size": 2048,
    "topks": "5,10",
    "beta_candidates": "0,0.01,0.02,0.05,0.1,0.2"
  },
  "datasets": [
    {
      "name": "Clothing",
      "data_path": "./data/Clothing",
      "inter_file": "clothing.inter",
      "text_feat_file": "text_feat.npy",
      "image_feat_file": "image_feat.npy"
    },
    {
      "name": "Sports",
      "data_path": "./data/Sports",
      "inter_file": "sports.inter",
      "text_feat_file": "text_feat.npy",
      "image_feat_file": "image_feat.npy"
    },
    {
      "name": "Baby",
      "data_path": "./data/Baby",
      "inter_file": "baby.inter",
      "text_feat_file": "text_feat.npy",
      "image_feat_file": "image_feat.npy"
    }
  ]
}
```

## Sports and Baby

The same commands can be applied by changing the dataset arguments.

Sports:

```bash
--data_path ./data/Sports \
--inter_file sports.inter \
--dataset_name Sports
```

Baby:

```bash
--data_path ./data/Baby \
--inter_file baby.inter \
--dataset_name Baby
```

## Customization

### Text-only modality

```bash
--image_feat_file ""
```

### Accuracy-oriented beta selection

```bash
--beta_selection accuracy
```

### Accuracy-popularity trade-off selection

```bash
--beta_selection tradeoff \
--pop_penalty 0.20
```

Increasing `--pop_penalty` gives stronger preference to beta values that lower validation ARP, while the selected score still includes validation NDCG. This parameter should be chosen on the validation split and reported transparently.

### Different Head/Middle/Tail definitions

Use:

```bash
--tail_quantile <value> \
--head_quantile <value>
```

The final thresholds must match those stated in the manuscript.

## What the Additional Scripts Reproduce

The specialized scripts cover experiments involving the MPRC model itself:

- Raw-versus-Calibrated comparison;
- representation ablations;
- calibration ablations;
- beta sensitivity;
- Low/Medium/High user groups;
- Head/Middle/Tail item groups;
- repeated random seeds;
- qualitative case studies.

## Contact

For questions or issues, open an issue in the repository or contact the corresponding author.

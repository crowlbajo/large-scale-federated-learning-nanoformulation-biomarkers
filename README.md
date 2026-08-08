# Large-scale federated learning across diverse clinical networks reveals nanoformulation-specific digital biomarkers

FedICBS combines frozen multimodal encoders, patient–drug interaction features, site-local generalized linear models, inverse-variance aggregation, and cross-site invariance screening for six-month tuberculosis treatment-response prediction. The primary analysis uses 19,150 eligible TB Portals v8.2 records partitioned into nine country clients and evaluates five fixed random seeds.

## Installation

The reported software environment uses Python 3.10, PyTorch 2.3, CUDA 12.1, Flower 1.7, RDKit 2024.3, scikit-learn 1.5, and statsmodels 0.14.

```bash
conda env create -f environment.yml
conda activate fedicbs
pip install -e .
```

The container image is built with:

```bash
docker build -t fedicbs:1.0 .
```

## Data

Dataset access locations and licenses are listed in `datasets.txt`. TB Portals v8.2 is a registered-user public release governed by its Data Use Agreement. Patient-level files are never bundled with this project. TBX11K, Montgomery County, and Shenzhen radiographs support only the auxiliary imaging-representation probe because their labels describe TB presence rather than treatment response. ChEMBL v34 supplies drug structures and molecular descriptors.

Prepare a locally authorized TB Portals export containing the schema required by `fedicbs.data.records`:

```bash
fedicbs-prepare --config configs/main.yaml --records data/tb_portals_v8_2.parquet --output data/prepared --override experiment.batch_size=4096 --override experiment.learning_rate=0.000001 --override experiment.optimizer=adamw --override experiment.scheduler=cosine --override experiment.warmup_steps=1000 --override experiment.weight_decay=0.01 --override experiment.precision=bf16
```

The preparation command filters sites below 80 eligible cases, writes a Parquet cohort, and records SHA-256 digests in a local manifest. Expected disk use depends on the authorized imaging export; the tabular manifest is small, while radiographs can require hundreds of gigabytes.

## Experiment configuration

The paper reports 89 communication rounds, nine clients, five seeds, 128 candidate features, 21 selected invariant features, a 20% site-stratified test fraction, 5,000 bootstrap replicates, and production privacy parameters ε=5 and δ=1e-5. It does not report batch size, learning rate, optimizer, scheduler, warmup, weight decay, gradient accumulation, or numeric precision. These fields are deliberately unset in `configs/main.yaml`; the command refuses to run until explicit values are supplied. Values shown in the example command are operational inputs, not manuscript-reported settings, and must not be used to claim the published measurements.

## Training

The primary configuration is validated with:

```bash
fedicbs-train --config configs/main.yaml --data-manifest data/prepared/manifest.json --output outputs/main --override experiment.batch_size=4096 --override experiment.learning_rate=0.000001 --override experiment.optimizer=adamw --override experiment.scheduler=cosine --override experiment.warmup_steps=1000 --override experiment.weight_decay=0.01 --override experiment.precision=bf16
```

FedICBS runs in three phases: federated multimodal feature generation, site-local marginal logistic regression with sufficient-statistic exchange, and invariant-feature-restricted federated prediction. The screen uses α/128, a minimum of 80 records per site, pooled coefficient significance, and a Cochran Q critical value with K−1 degrees of freedom. Only coefficients, standard errors, covariance estimates, and counts enter the invariance coordinator.

The reported hardware allocation is eight NVIDIA A100 80GB GPUs per simulated client. With nine clients and five seeds, the reported total is approximately 630 GPU-hours. Storage must accommodate the registered TB Portals export, cached frozen-encoder embeddings, prepared tables, and run artifacts.

## Evaluation

Evaluate a CSV containing `label` and `probability` columns:

```bash
fedicbs-evaluate --config configs/main.yaml --predictions outputs/main/predictions.csv --output outputs/main/evaluation --override experiment.batch_size=4096 --override experiment.learning_rate=0.000001 --override experiment.optimizer=adamw --override experiment.scheduler=cosine --override experiment.warmup_steps=1000 --override experiment.weight_decay=0.01 --override experiment.precision=bf16
```

The evaluation package provides AUROC, sensitivity, specificity, F1, stratified bootstrap confidence intervals, DeLong comparison, two-way mixed-effects ICC, Cochran–Mantel–Haenszel subgroup analysis, equalized-odds difference, site spread, and leave-one-country-out summaries. The manuscript target for the five-seed primary analysis is AUROC 0.946 ± 0.008, sensitivity 0.912 ± 0.015, specificity 0.934 ± 0.011, and F1 0.921 ± 0.012. These values are acceptance references for an authorized complete cohort, not guarantees for a differently filtered export or unreported optimization settings.

## Package layout

`code/fedicbs/data` handles registered cohort tables, site-local preprocessing, radiographs, and encoded records. `code/fedicbs/models` contains modality projections, the four-layer tabular encoder, eight-head four-layer Perceiver fusion, the 128-dimensional candidate assembler, and invariant prediction head. `code/fedicbs/science` contains nanoformulation kinetics, marginal invariance testing, and privacy mechanisms. `code/fedicbs/federation` contains aggregation, comparator strategies, and the three-phase coordinator. `code/fedicbs/training` contains optimization, mixed precision, distributed training support, loss functions, seed control, and atomic state persistence. `code/fedicbs/evaluation` contains metrics, statistical comparisons, LOCO analysis, attribution, and result serialization.

## Privacy and data governance

No patient identifiers, source records, credentials, access tokens, or local machine locations belong in configuration committed to version control. Each participating site performs imputation and standardization from its own training fold. Outputs should use study identifiers that cannot be mapped back to clinical records outside the authorized environment. Differential privacy accounting and sufficient-statistic transport do not replace the TB Portals Data Use Agreement or local institutional controls.

## License

The software is provided under the MIT License. Dataset licenses and access agreements remain independent and are summarized in `datasets.txt`.

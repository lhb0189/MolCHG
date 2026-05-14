# MolCHG
## Data Preparation

The `.pt` files required for pre-training and fine-tuning are **not included** 
in this repository due to file size limits. You can generate them locally from 
the raw CSV datasets using the provided script.

### Step 1: Convert CSV to .pt

Run `csv_to_pt.py` to convert raw CSV datasets into the `.pt` format used by 
the model:

```bash
python csv_to_pt.py
```

This script reads the CSV files (e.g., from `Pretrain_Datasets/zinc/` for 
pre-training or `Finetune_Datasets/` for downstream tasks) and produces the 
corresponding `.pt` files, which contain the processed molecular graphs ready 
to be fed into the model.

### Step 2: Run Pre-training or Fine-tuning

Once the `.pt` files are generated, you can launch training by simply updating 
the dataset file paths inside the scripts:

- **Pre-training**: open `pretrain.py` and set the input filename to your 
  generated pre-training `.pt` file, then run:
```bash
  python pretrain.py
```

- **Fine-tuning**: open `finetune_train.py` and set the input filename to your 
  generated fine-tuning `.pt` file, then run:
```bash
  python finetune_train.py
```

No additional configuration is required — modifying the file names in these 
two scripts is sufficient to reproduce the experiments.

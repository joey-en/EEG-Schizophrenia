# EEG-Schizophrenia

Download only the `.csv` files from the Kaggle dataset `broach/button-tone-sz`.

## Run

Open `cmd.exe` in the repo root and run:

```cmd
conda activate DSAI4202
pip install -r requirements.txt
```

## Notebook Setup

If you will commit `.ipynb` files, install the notebook output stripping hook once in this repo:

```cmd
nbstripout --install
```

The repo already includes `.gitattributes` so `*.ipynb` files use the `nbstripout` filter.

For notebook work, prefer launching Jupyter from the CLI in the repo root and continuing to edit in Jupyter rather than opening the notebook directly in VS Code. That keeps the notebook kernel aligned with the same environment you use in the terminal.

Example:

```cmd
conda activate DSAI4202
jupyter notebook
```

Open the `http://localhost:...` link printed in the terminal and work from there.

If you really want to use VS Code for the notebook UI, connect VS Code to the same local Jupyter server started from the CLI and make sure the notebook kernel/interpreter is set to `pyspark 4 (conda)`.

## Dataset Download

After setup, configure your Kaggle token and run:

```cmd
set KAGGLE_API_TOKEN=your_token_here
python src/downloadData.py
```

Kaggle API token setup instructions:

- Official Kaggle auth guidance: https://github.com/Kaggle/kagglehub#authenticate

## What It Does

- Lists the files in the Kaggle dataset first
- Downloads only `.csv` files
- Skips `.tar` and other non-CSV files entirely
- Preserves nested Kaggle folders under `data/raw`
- Prints what is downloading and what is skipped

## Notes

- The `set KAGGLE_API_TOKEN=...` command is for `cmd.exe`
- `nbstripout --install` is a one-time repo setup step, not something you need before every download
- If you use PowerShell instead, use:

```powershell
$env:KAGGLE_API_TOKEN="your_token_here"
python src/downloadData.py
```

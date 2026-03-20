# EEG-Schizophrenia

This project works with the Kaggle dataset `broach/button-tone-sz` and prepares the data for notebook analysis. To understand the project and the workflow, start with the data reduction notebook at `notebooks/1_data_reduction.ipynb`.

## Initial Setup

### Environment Setup

Open a terminal in the repo root and run:

```powershell
conda activate DSAI4202
pip install -e ".[all]"
nbstripout --install
```

'`pip install -e ".[all]"` is similar to `pip install -r requirements.txt`

`nbstripout` removes notebook output before commits so `.ipynb` files stay smaller and cleaner in Git. The repo already includes `.gitattributes` so notebook files use the `nbstripout` filter after you install it once in your local clone.

### Dataset Download

Before you run the downloader, get your Kaggle API token so the command does not fail on authentication.

1. Sign in to Kaggle.
2. Open `Settings`.
3. Go to the `API` section.
4. Click `Generate New Token`.
5. Copy the token value you want to use for `KAGGLE_API_TOKEN`.

Full Official Kaggle auth guidance here: https://github.com/Kaggle/kagglehub#authenticate

After setup, configure your Kaggle token and download the dataset:

```powershell
# Command Prompt
setx KAGGLE_API_TOKEN "your_token_here"
download_eeg_schizophrenia

# PowerShell
$env:KAGGLE_API_TOKEN="your_token_here"
download_eeg_schizophrenia
```

`setx` stores the variable permanently for future terminals. After running it, close and reopen the terminal so the persisted value is available.

## Start Coding

Everytime you want to code, start Jupyter from the repo root with the same environment you use in the terminal:

```powershell
conda activate DSAI4202
jupyter notebook
```

Open the `http://localhost:...` link printed in the terminal and edit the notebook in Jupyter.

If you really want to use VS Code for the notebook UI, copy that same Jupyter link, then in VS Code select the kernel picker, choose `Existing Jupyter Server`, and paste the remote URL there.

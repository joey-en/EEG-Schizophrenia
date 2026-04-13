# EEG-Based Schizophrenia Classification with PySpark

## Project Overview

This project builds an EEG-based classification pipeline to distinguish schizophrenia cases from control subjects. The goal is not to replace clinical diagnosis, but to explore whether brain-response patterns in auditory tasks can support earlier screening and decision support.

The project is centered on two ideas:

- Use **PySpark** to process large raw EEG recordings efficiently.
- Use **domain-driven feature engineering**, especially **N100 suppression**, to turn weak raw signals into more informative modeling features.

At a high level, the workflow starts with raw EEG CSV files from Kaggle, reduces them to trial-level ERP summaries, engineers clinically motivated features, preprocesses them with Spark ML, applies PCA for dimensionality reduction, and trains classification models in Spark.

## Dataset Description

The source data comes from the Kaggle dataset [`broach/button-tone-sz`](https://www.kaggle.com/datasets/broach/button-tone-sz). The raw data contains roughly **20 GB** of EEG time-series recordings from **81 participants** across **3 task conditions**, with repeated trials and sample-level brain activity measurements.

After data reduction, the working table becomes a **trial-level dataset**:

- Each row represents **one trial for one subject under one condition**.
- Key identifiers include `subject`, `trial`, and `condition`.
- Key signal features include ERP summaries such as `Fz_N100_avg`, `FCz_P200_avg`, and other electrode- and region-level measures.
- A `label` column identifies the class: control vs schizophrenia.

### Simple EEG Terms Used in This Project

- **N100**: an early negative brain response that appears about 100 ms after a sound.
- **P200**: a later positive brain response that appears about 200 ms after a sound.
- **Active condition**: the participant is involved in generating or expecting the sound.
- **Passive condition**: the sound is externally presented.
- **Control condition**: a third task condition included in the dataset and retained in the pipeline.

These components matter because schizophrenia is often associated with differences in how the brain processes expected vs external sensory events.

## Workflow Overview

The project follows the notebook order below:

1. `notebooks/1_data_reduction.ipynb`
2. `notebooks/2_eda.ipynb`
3. `notebooks/3_feature_engineering.ipynb`
4. `notebooks/4_data_preprocessing.ipynb`
5. `notebooks/5_modeling.ipynb`

### End-to-End Pipeline

- **Data ingestion**: download raw EEG CSV files from Kaggle and load them with Spark.
- **Data reduction**: convert sample-level EEG signals into trial-level summaries using aggregation, ERP extraction, and SAX-based symbolic summaries.
- **EDA**: inspect class balance, demographic context, ERP behavior, and condition effects.
- **Feature engineering**: create regional, asymmetry, baseline-adjusted, suppression, and variability features.
- **Preprocessing**: join labels, encode categorical fields, assemble features, scale them, and create PCA features.
- **Modeling**: train binary classifiers on the PCA-based feature vectors.
- **Evaluation**: compare models using recall, precision, and PR-AUC.

```text
[Kaggle EEG CSVs]
    -> [PySpark ingestion]
    -> [ERP + SAX reduction]
    -> [Reduced trial table]
    -> [EDA + label integration]
    -> [Feature engineering]
    -> [Engineered modeling table]

[Engineered modeling table]
    -> [Spark preprocessing]
    -> [PCA on combined reduced raw + engineered features]
    -> [PCA modeling dataset]
    -> [Classification models]
    -> [Evaluation: Recall, Precision, PR-AUC]
    -> [Key insight: suppression and condition-based features outperform raw ERP amplitudes]
```

## How PySpark Was Used

PySpark is a core part of the project, not just a preprocessing helper.

### Where Spark Was Used

- Reading large raw EEG files and processed parquet datasets
- Joining EEG recordings with time maps and demographic labels
- Trial-level feature extraction with `groupBy(...).agg(...)` using functions such as `AVG`, `MAX`, `MIN`, and `STDDEV`
- ERP window aggregation using Spark SQL / DataFrame logic
- Window-based symbolic reduction using SQL functions such as `NTILE`
- Preprocessing with Spark ML components:
  - `StringIndexer`
  - `OneHotEncoder`
  - `VectorAssembler`
  - `StandardScaler`
  - `PCA`
- Model training with Spark ML classifiers

### Why Spark Was Necessary

The raw EEG data is large and highly granular. Each participant contributes many trials, and each trial contains thousands of signal samples. Operations such as grouping by subject/trial/condition, extracting ERP windows, and computing many electrode-level summaries are expensive to manage in a purely pandas-based workflow. Spark makes those reductions and joins more practical and scalable.

### What Was Not Spark

The main non-Spark pieces were:

- Plotly and Matplotlib visualizations
- Small pandas conversions for EDA plots and explained-variance inspection

Unlike many mixed pipelines, the modeling stage here was also implemented with **Spark ML**.

## Feature Engineering

Feature engineering was necessary because the EDA showed that **raw ERP amplitudes alone do not clearly separate the two classes**. The project therefore focuses on creating features that better reflect meaningful brain-response patterns.

### Main Feature Groups

- **Regional features**: electrode values were grouped into interpretable brain regions such as frontal, central, frontocentral, parietal, left, right, and global summaries.
- **Baseline-adjusted features**: N100 and P200 values were adjusted relative to the pre-stimulus baseline (`B0`) to reduce offset effects and better capture response changes.
- **Asymmetry features**: left-vs-right differences such as `FC3 - FC4`, `C3 - C4`, and `CP3 - CP4` were created to capture hemispheric imbalance.
- **Suppression features**: subject-level N100 suppression was computed from **Passive - Active** responses across electrodes and regions.
- **Variability features**: trial-level standard deviation, range, and coefficient of variation were added to capture response consistency.

### Why N100 Suppression Matters

N100 suppression is the most important engineered idea in this project.

In simple terms, healthy brains usually reduce their response to a self-generated or expected sound. That means the **Active** response should be weaker than the **Passive** response. This difference is called **suppression**.

In the EDA, control subjects showed a clearer suppression pattern, while schizophrenia subjects showed a smaller and less consistent difference between Active and Passive conditions. That makes suppression a useful feature for classification because it captures a meaningful physiological difference rather than relying on raw amplitude alone.

## Preprocessing

The preprocessing stage prepared the engineered trial-level table for modeling.

- Duplicate rows were removed where necessary.
- Demographic labels were joined to the modeling table.
- Categorical fields were encoded with Spark ML.
- Numeric and encoded features were assembled into vectors with `VectorAssembler`.
- Feature vectors were standardized with `StandardScaler`.
- A Spark PCA step was applied to the combined reduced raw + engineered feature space.

Important clarification: **PCA is dimensionality reduction, not original-feature selection.** It creates new principal components from the combined feature set rather than selecting a literal subset of the original columns.

The train/validation/test split was performed at the **subject level** to avoid leakage between trials from the same person. Baseline model comparison and hyperparameter tuning were carried out on the validation split, while the final held-out test split was reserved for the selected Random Forest model.

## Modeling

The modeling notebook uses the PCA-based dataset and trains several binary classifiers in Spark:

- **Logistic Regression**: interpretable linear baseline with `elasticNetParam=0.5`
- **Decision Tree**: simple nonlinear baseline
- **Random Forest**: stronger nonlinear ensemble for interactions across features
- **Linear SVC**: large-margin linear classifier with `maxIter=100` and `regParam=0.1`
- **Gaussian Naive Bayes**: lightweight probabilistic benchmark
- **Multilayer Perceptron**: nonlinear neural baseline with layers `[input, 64, 32, 2]` and `maxIter=100`

Random Forest was later selected for additional tuning because it delivered the strongest recall, which was the primary metric in this project.

## Evaluation Metrics

The project evaluates models through a screening-oriented lens, where missing a schizophrenia case is more costly than flagging an extra case for follow-up.

- **Precision**: of the cases predicted as schizophrenia, how many were correct
- **Recall**: of the true schizophrenia cases, how many were correctly identified
- **PR-AUC**: summarizes the precision-recall tradeoff across thresholds

PR-AUC is especially useful here because the classes are not perfectly balanced and the positive class is the one of greatest practical interest. It gives a better picture of positive-case performance than accuracy alone.

## Results

The table below summarizes the baseline model comparison on the **validation set** from the modeling notebook.

| Model | Recall | Precision | PR-AUC |
|------|--------|-----------|--------|
| **Random Forest** | **0.594** | 0.495 | 0.666 |
| Logistic Regression | 0.558 | **0.606** | **0.734** |
| Decision Tree | 0.561 | 0.567 | 0.686 |
| Multilayer Perceptron | 0.564 | 0.579 | 0.651 |
| Linear SVC | 0.527 | 0.560 | 0.658 |
| Naive Bayes | 0.456 | 0.488 | 0.665 |

**Best validation model:** Random Forest, because it achieved the highest recall, which was the primary selection criterion.

**Best validation PR-AUC:** Logistic Regression, indicating stronger ranking performance even though its recall was lower than Random Forest.

For the final held-out evaluation, the selected Random Forest model achieved:

- **Train recall:** 0.704
- **Train precision:** 0.778
- **Train PR-AUC:** 0.909
- **Test recall:** 0.495
- **Test precision:** 0.350
- **Test PR-AUC:** 0.509

This gap between training and test performance suggests that the selected Random Forest model captures useful signal, but still shows signs of overfitting on unseen subjects.

## Key Insights

- **Raw ERP amplitudes alone were not strongly discriminative.** The control and schizophrenia groups showed substantial overlap in electrode-level ERP plots.
- **Suppression features were more informative.** Active vs Passive response differences captured a clearer physiological distinction between groups.
- **Variability and condition-based features mattered.** Schizophrenia responses appeared less consistent, which made variability statistics useful alongside amplitude summaries.
- **Feature engineering was more valuable than relying on raw summaries alone.** The strongest story in the project comes from transforming EEG signals into more meaningful condition-aware features.

## Recommendations for Stakeholders

- This approach is promising as a **clinical decision-support or screening aid**, especially because it uses non-invasive EEG data and captures interpretable response patterns.
- The feature design is grounded in neuroscience, which makes the outputs easier to justify than a purely black-box signal approach.
- The project should **not** be treated as a standalone diagnostic tool. It is better suited to prioritization, screening, or supporting follow-up assessment.

### Current Limitations

- The dataset is relatively small at the subject level.
- EEG signals are noisy and highly variable across people and trials.
- Local hardware constraints limited the scope of cross-validation and tuning.
- External validation on a separate cohort has not yet been demonstrated in this workflow.

### Future Improvements

- Evaluate on larger and independent EEG cohorts
- Improve calibration and threshold selection for screening use
- Compare PCA-based modeling against alternative dimensionality-reduction strategies
- Add model explainability and clinician-friendly summaries
- Explore richer temporal modeling while preserving the current Spark-based preprocessing advantages

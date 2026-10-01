# Honey Spectral Data Fusion
Classifying the botanical and geographical origin of honey from FTIR and UV-Vis spectra using Random Forest and data fusion at the data, feature, and decision levels.


# Overview
Spectroscopy produces rich but high-dimensional data, and a single technique often captures only part of the chemical information in a sample. This project investigates whether combining two spectroscopic sources improves the classification of honey samples.

Two datasets of honey samples were used: one from UV-Vis spectroscopy and one from FTIR spectroscopy. The goal was to build the best possible model for predicting the botanical and geographical origin of each sample.

The experiments were run in two stages:

Baselines: a Random Forest classifier trained on each dataset separately.
Data fusion: three fusion strategies (data-level, feature-level, decision-level) combining both sources.

Key finding: every fusion level outperformed the single-source baselines, and decision-level fusion gave the best results (87% geographical and 90% botanical accuracy).

# Datasets
| Dataset | Technique | Description |
|---------|-----------|-------------|
| `ftir.csv` | FTIR (Fourier-Transform Infrared Spectroscopy) | Spectral measurements of the absorption of infrared radiation by honey samples, with geographical and botanical origin labels |
| `uvvis.csv` | UV-Vis (Ultraviolet-Visible Spectroscopy) | Spectral measurements of the absorption of ultraviolet and visible radiation by honey samples, with geographical and botanical origin labels |

## Data Availability
The datasets were collected from a honey production facility for a PhD thesis and shared with me by my professor for this experiment. They are not publicly available and are therefore not included in this repository.

# Methodology
All experiments share the same base pipeline:

1)Load and preprocess: remove non-informative columns and separate features from labels.

2)Normalize: standardize features so that all have the same scale.

3)Dimensionality reduction: Principal Component Analysis (PCA), keeping 95% of the variance.

4)Split: divide the data into training and test sets.

5)Classify: train a Random Forest classifier, separately for the geographical and the botanical target.

6)Evaluate: report accuracy and the classification report (including F1-score) on the test set.

# Data Fusion Techniques
Data-level fusion

The two datasets are loaded and merged on a common key column into a single dataset. The standard pipeline (preprocessing, normalization, split, Random Forest) is then applied to the merged data.

Feature-level fusion

Both datasets are loaded, preprocessed, and aligned so that samples match. Each dataset is normalized and reduced with PCA separately. The resulting PCA features are concatenated into one feature set, which is used to train the Random Forest.

Decision-level fusion

One Random Forest is trained per dataset, for each target. Instead of using the models' hard predictions, their predicted class probabilities are averaged (probability averaging), and the class with the highest mean probability is chosen as the final prediction. PCA is not required here, since the focus is on combining decisions rather than optimizing features.

# Observations

FTIR alone is clearly more informative than UV-Vis alone, especially for botanical origin.
Data-level fusion already improves on both single sources, most notably for botanical origin (59% to 83% compared with the best baseline).
Feature-level fusion improves further over data-level fusion.
Decision-level fusion achieves the best overall performance, showing that combining the outputs of separate models is a strong approach for spectroscopic analysis.

Full table: results/results_table.md

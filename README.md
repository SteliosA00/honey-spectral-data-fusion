# honey-spectral-data-fusion
Classifying honey botanical and geographical origin from FTIR and UV-Vis spectra using Random Forest and data fusion at the data, feature, and decision levels.


This project explores how data fusion can improve machine learning performance on spectral data. Using two datasets of honey samples, one from FTIR spectroscopy and one from UV-Vis spectroscopy, I first trained baseline Random Forest classifiers on each dataset separately to predict botanical and geographical origin. I then applied three fusion strategies: data-level (merging the raw datasets), feature-level (concatenating PCA-reduced features), and decision-level (averaging the predicted class probabilities of the two models). All fusion approaches outperformed the single-source baselines, with decision-level fusion achieving the best results (87% geographical and 90% botanical accuracy).

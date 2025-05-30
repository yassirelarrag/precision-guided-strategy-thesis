# 📈 Stock Movement Prediction Using Random Forests – Thesis Codebase

This repository contains the full codebase, prediction files, and processed data used in the thesis.

## 📂 Repository Structure
This repository contains all code, data, and supporting files used for the thesis. Below is a description of the key contents:

Data/
Contains the historical stock and index data used in the study:

historical_data_AAPL.csv: Historical data for Apple Inc. (AAPL)

historical_data_BA.csv: Historical data for Boeing Co. (BA)

historical_data_^GSPC.csv: Historical data for the S&P 500 index

env/
The virtual environment directory (not tracked in Git). Can be recreated using requirements.txt.

model_prediction_AAPL.pkl
Precomputed model predictions for AAPL stock.

model_prediction_BA.pkl
Precomputed model predictions for BA stock.

selected_features_AAPL.pkl
Selected features used for training the AAPL model.

selected_features_BA.pkl
Selected features used for training the BA model.

analysis.ipynb
Main Jupyter notebook containing all code for data engineering, model training, threshold tuning, and simulation.

engineering.py
Python script containing functions for data processing and technical indicator computation.

Plots/
Folder with visual outputs of the simulations and analysis:

requirements.txt
A list of required Python packages to recreate the environment.

README.md
Documentation explaining the purpose of the project and how to use it.

## Interpretable RL Framework for ICU Hypotension Management

This repo contains notebooks, code and files associated with paper `An Interpretable RL Framework for Pre-deployment Case Review, Improvement, and Validation Applied to Hypotension Management in the ICU`

### Requirements
- python 3.7 or above
- pytorch version 1.10 or above
- numpy 1.21 or above
- sklearn 1.0.1
- pandas 1.3.4

### Code Access

- `notebook`: the folder contains jupyter notebooks required to produce our research results. To reproduce the results in our manuscript, you can execute the ordered notebooks. Note, before running `03.identify_and_evaluate_cluster.ipynb`, you need to run the code `launch_P_calc_four_action.py` to determine which points among the train and test data are decision points. 
- `src`: the folder contains helper functions.
- `results`: the folder contains intermediate, visualizations and evaluation results from running the notebooks.

### Data Access
The data we used for this project was derived from [MIMIC-III - Medical Information Mart for Intensive Care](https://registry.opendata.aws/mimiciii/). To access data, please contact Dr.Finale Doshi who runs Data to Actionable Knowledge (DtAK) Lab at finale@seas.harvard.edu. 
# capstoneml
FPGA Bitstreams are required to run certain scripts on the Ultra96. The bitstreams were intentionally left out of the repo as since users can already be compiled from the code in this repo.

## notebook
In notebooks, we have the notebooks used to train and evaluate models on both local computer and on the ultra96 board.

### Training Model:
- conv_trainer.ipynb
- explore_conv.ipynb: Used during exploration of a CNN model
- main.ipynb: Training model on Unseen Data

### Feature Extraction:
- feature_playground.ipynb: Playground to explore feature extraction
- feature_emd.ipynb: Exploration of more features

### Evaluations:
- main_clean.ipynb: locally evaluate and test out algorithm
- eval_conv.ipynb: Evaluate and test algorithm with FPGA
- video.ipynb: Notebook used during Idividual Progress Check

## Scripts

### Main Scripts on Ultra96
### Actual Implementation:
- ai.py
- start_detector.py
- modelling_utils.py
- filepaths_extComms.py
- dataloader.py

### All Notebooks
- cpu_implement.py
- dataloader_notebook.py
- filepaths.py
- filereader.py
- model_utils_training.py
- visualisers.py

### Ultra96 Notebook Exclusive Use:
- ai_notebook.py
- start_detector_notebook.py

### Local Notebook Use:
- ai_tensor.py
- start_detector_tensor.py

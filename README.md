
# Automatic classification of chains of guitar effects through evolutionary neural architecture search

Listen to the samples of our new dataset here: https://michelerossi1.github.io/guitar_dataset_examples/

![paper_effects](https://user-images.githubusercontent.com/61735529/232543661-e4938fa6-b574-4799-97c3-b3639ec8aacb.png)

# Repository Overview  

This repository contains the following files and folders:  

### dataset_creation  
This folder contains the code used to create the datasets for the paper *"Automatic Classification of Chains of Guitar Effects through Evolutionary Neural Architecture Search"*. The clean dataset was processed using VST plugins or the Pedalboard library. The folder includes four `.py` files: three for creating datasets of effects chains (for multi-label classification) and one for creating a dataset of single effects (for multiclass classification, which is not included in the paper). This last file covers a wider range of guitar effects compared to those used for the chains.  

### 1__paper_guitar_effects  
This `.py` file includes the primary code for the paper. It was written and executed in Google Colab Pro+, following the block-based structure typical of Jupyter Notebook environments. The code is comprehensive and covers preprocessing, deep learning classification functions, the implementation of Neural Architecture Search (NAS) based on genetic algorithms, a comparison with random search, the implementation of the best models found, and the computation of statistics, among other features. Extensive comments have been added to enhance readability.  

### 2__plot  
This file includes the code for generating plots, such as bar plots, confusion matrices, and Pareto fronts, as well as code used to investigate the best architectures. The plots displayed in this file may not necessarily represent the final results but are part of the experimental investigations (refer to the paper for finalized plots). Additionally, the paper used a dedicated package named `fpgplots` for displaying the results instead of Matplotlib.  

If you have any questions about the code or materials, feel free to contact me at [michele.rossi-2@unitn.it](mailto:michele.rossi-2@unitn.it).  



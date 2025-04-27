# INM705 Deep Learning for Image Analysis - Coursework

## Overview
For this project, we used Google Colab Pro to leverage L4 and A100 GPUs for accelerated training. Our main goal was to compare two different architectures for crime video classification:

* **3D CNN**
* **DINOv2 + Transformer**

The project is structured around two main notebooks, each dedicated to one of these architectures. All fine-tuning experiments, model phases, and results are contained within these respective notebooks (filenames provided below).

### Main Notebooks:
* `MAIN_3DCNN.ipynb`
* `MAIN_DINOv2_Transformer.ipynb` 

Each notebook also includes:

* Detailed markdowns explaining the architecture setup, experimental phases, hyperparameter tuning, and result interpretations.
* Links to corresponding Weights & Biases (wandb) project logs for further exploration. If you encounter any issues accessing the wandb logs, feel free to reach out and we’ll assist with access.

#### Notebook Viewing Note

If you encounter the following error while trying to view the Jupyter Notebooks (`.ipynb`) on GitHub:

*Invalid Notebook: There was an error rendering your Notebook: the 'state' key is missing from 'metadata.widgets'. Add 'state' to each, or remove 'metadata.widgets'.*

This is a **GitHub rendering issue** related to Jupyter widget metadata. It does **not affect the actual notebook functionality**.

> **Solution:**  
Please **download the notebook** and open it using any Jupyter-supported IDE.  
The notebook should work as expected without any issues.

## Dataset Setup
Before running any code:
1. **Download the dataset zip file from the following link: https://drive.google.com/file/d/145OHbfMAaot25bgx2tDRsiIXvOKLvezi/view?usp=sharing**
2. **Upload the extracted folder to your Google Drive.**

Our code is specifically tailored to Colab, with data paths pointing to Drive locations.

Note: The dataset is a custom-trimmed version of the original UCF-Crime dataset (retrieved from Kaggle). It contains preprocessed and shortened data that we used for our experiments.

## Additional Files
We’ve included two extra notebooks for reference:
* `EXTRA - Comparison of Other Architectures.ipynb` 

  Contains our early experimentation with various architectures that helped guide our final selection of the two main models compared in this project.

  **You don’t need to run this file.**

* `EXTRA - Dataset Trim.ipynb`

  Shows the process we followed to trim and preprocess the original UCF-Crime dataset into the custom version we used for this project.

  **You don’t need to run this file either.**

These files are just there for transparency and completeness.




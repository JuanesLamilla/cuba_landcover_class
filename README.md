# Cuba Land Cover Classification Model Testing

## Introduction

This repository contains code for a land cover classification analysis for Cuba using Landsat 8 imagery. 4 different models are created, optimized, tested, and compared. The models are:

1. CART (Decision Tree)
2. Random Forest
3. XGBoost
4. Neural Network

## Installation

To install environment dependencies using conda, run the following command in the terminal:

```bash
conda env create --name cuba_classification --file=environment.yml
```

Then to activate the environment you can use:

```bash
conda activate cuba_classification
```

You can see independent packages and their versions in the `environment.yml` file.

You can now use the `cuba_classification` environment to run the code in the `analysis.ipynb` notebook.

## Usage

To comply with assignment guidelines, the entirety of the code is in the `analysis.ipynb` notebook.

To run the code, you will need to a Google Earth Engine account. When the notebook is first run, it will open a browser window asking you to authenticate your account. Once you have done this, it will give you an access token which you can copy and paste into the notebook (where it requests it). You only need to do this once.

Due to long runtimes during model training and preprocessing, the various models and some of the preprocessed data is saved in the `models` and `temporary` folders. If you want to re-run *all* code, you can delete these folders and re-run the notebook. **Note: This will take multiple hours**. If you want to re-run only parts of the code, you can delete the specific files you want to re-calculate.





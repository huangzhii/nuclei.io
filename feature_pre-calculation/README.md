# nuclei.io feature pre-calculation

This software is an open-release under the study of **A pathologist–AI collaboration framework for enhancing diagnostic accuracies and efficiencies** published in **Nature Biomedical Engineering**.

For the full information, please visit our homepage: https://huangzhii.github.io/nuclei-HAI/

If you are a pathologist/user/developer who plan to use this software for annotating / analyzing whole slide pathology images, please follow the tutorial below.

We are also releasing a Youtube tutorial soon. Please stay tuned!

## Pre-requisite
Please install openslide on your computer: https://openslide.org/download/

## Step 1. Download an example whole slide image.
```bash
cd nuclei.io/
mkdir example_data;
cd example_data;
mkdir CMU_Aperio;
wget -O CMU_Aperio/CMU-1.svs https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs
wget -O CMU_Aperio/CMU-2.svs https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-2.svs


## Initialization
We use Anaconda to manage the codes.

Note: If you do not have Anaconda yet, please follow this tutorial to install anaconda on your computer: https://docs.anaconda.com/free/anaconda/install/index.html

Step 1. Initialize anaconda environment.
```bash
conda create -n nuclei-feature python=3.10
conda activate nuclei-feature
```

## Step 2. Install dependencies
```bash
pip install --upgrade pip  # enable PEP 660 support
conda install openslide -c conda-forge # install openslide using conda.
pip install -e .
```

## Step 3. Run whole slide image cell segmentation pipeline
```python
python main.py --slidepath "../example_data/CMU_Aperio/CMU-1/CMU-1.svs" \
     --stardist_dir "../example_data/CMU_Aperio/CMU-1/stardist_results" \
     --stage "segmentation"
```

## Step 4. Run nuclei feature calculation pipeline
```python
python main.py --slidepath "../example_data/CMU_Aperio/CMU-1/CMU-1.svs" \
     --stardist_dir "../example_data/CMU_Aperio/CMU-1/stardist_results" \
     --stage "feature"
```

## Step 4. Run whole slide image cell segmentation pipeline
```python
python main.py --slidepath "../example_data/CMU_Aperio/CMU-1/CMU-1.svs" \
     --stardist_dir "../example_data/CMU_Aperio/CMU-1/stardist_results" \
     --stage "deeplearning_feature"
```
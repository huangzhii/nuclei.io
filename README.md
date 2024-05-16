# nuclei.io: Human-in-the-loop active learning framework for pathology image analysis

## Read me
This software is an open-release under the study of **A pathologist–AI collaboration framework for enhancing diagnostic accuracies and efficiencies** published in **Nature Biomedical Engineering**.

For the full information, please visit our homepage: https://huangzhii.github.io/nuclei-HAI/

If you are a pathologist/user/developer who plan to use this software for annotating / analyzing whole slide pathology images, please follow the tutorial below.

We are also releasing a Youtube tutorial soon. Please stay tuned!


## Initialization
We use Anaconda to manage the codes.

Note: If you do not have Anaconda yet, please follow this tutorial to install anaconda on your computer: https://docs.anaconda.com/free/anaconda/install/index.html

Step 1. Initialize anaconda environment.
```bash
conda create -n nuclei.io python=3.10
conda activate nuclei.io
# Install dependencies
conda install openslide -c conda-forge # install openslide using conda.
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

## Optional: Use nuclei.io feature pre-calculation pipeline to extract nuclei information.


### Step 1. Download an example whole slide image.
```bash
cd nuclei.io/
mkdir example_data;
cd example_data;
mkdir CMU_Aperio;
mkdir CMU_Aperio/CMU-1;
mkdir CMU_Aperio/CMU-2;
wget -O CMU_Aperio/CMU-1/CMU-1.svs https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs
wget -O CMU_Aperio/CMU-2/CMU-2.svs https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-2.svs
```

### Step 2. Run feature pre-calculation code for `CMU-1.svs`.
```python
python feature_pre-calculation/main.py --slidepath "example_data/CMU_Aperio/CMU-1/CMU-1.svs" \
     --stardist_dir "../example_data/CMU_Aperio/CMU-1/stardist_results" \
     --stage "all"

python feature_pre-calculation/main.py --slidepath "example_data/CMU_Aperio/CMU-2/CMU-2.svs" \
     --stardist_dir "../example_data/CMU_Aperio/CMU-2/stardist_results" \
     --stage "all"
```

## Start using nuclei.io to annotate and visualize your data.
```bash
python software/main.py
```
Now, on the sidebar, click "Browse local", and open the folder `example_data/CMU_Aperio/CMU-1/`.


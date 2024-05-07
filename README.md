# nuclei.io: Human-in-the-loop active learning framework for pathology image analysis

<img src="assets/images/under_construction_PNG42.png" alt="UNDER CONSTRUCTION" height="200">

Note: This website is under construction for final release. ETA: `5/20/2024`.


## Read me
This software is an open-release under the study of **A pathologist–AI collaboration framework for enhancing diagnostic accuracies and efficiencies** published in **Nature Biomedical Engineering**.

For the full information, please visit our homepage: https://huangzhii.github.io/nuclei-HAI/

If you are a pathologist/user/developer who plan to use this software for annotating / analyzing whole slide pathology images, please follow the tutorial below.

We are also releasing a Youtube tutorial soon. Please stay tuned!

## Download an example whole slide image.
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
conda create -n nuclei.io python=3.10
conda activate nuclei.io
```

Step 2. Install dependencies
```bash
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

Step 3. Open software
```bash
python software/main.py
```
Now, on the sidebar, click "Browse local", and open the folder `example_data/CMU_Aperio`.
# nuclei.io: Human-in-the-loop active learning framework for pathology image analysis

## Read me
This software is an open-release under the study of **A pathologist–AI collaboration framework for enhancing diagnostic accuracies and efficiencies** published in *Nature Biomedical Engineering*.

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
```

Step 2. Install dependencies
```bash
cd nuclei.io/
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

Step 3. Open software
```bash
python software/main.py
```
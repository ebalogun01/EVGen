# EVGen: Adversarial Networks for Learning Electric Vehicle Charging Loads and Hidden Representations

Paper (published at ICML 2021): https://arxiv.org/abs/2108.03762

Note: training data was proprietary unfortunately so we cannot give a completely reproducible repo. However, we have included trained models in this repo.

---

## Creating Conda Enviroment

For windows users: `conda env create -f EVGen.yml`

---
## Models
`EVGen/models/*` contains the trained models used to produce the paper.

---

## Writing results
Each training session will either make a new model or continue training an old model. 
Each training session will have its own dedicated results subdirectory under a main results directory. Each subdirectory contains:

1) A copy of the configuration file that was used to train the model.
2) A copy of the training and test data sets, saved as h5 files (this has been hidden due to data agreement).
3) Tensorboard log files and log directories (it is likely that tensorboard is unsupported). 
4) A models directory. Models are saved to disk in increments with size set in configuration.
5) An images directory. Sample outputs from the generator are plotted during training and saved as images to this directory.
6) An output file. This file captures the output to stderr and stdout during training.
---
## Files
- configs.json
    - JSON file containing hyperparameters and model training instructions
- entry_point_normal.py
    - Normal GAN
    - Uses weight clipping
- entry_point_WGAN.py
    - Wasserstein GAN
    - Uses Gradient Penalty
- entry_point_SCWGAN.py
    - Wasserstein GAN with similarity constraint
    - Uses Gradient Penalty
- gan.py
    - Contains the model architectures and forward instructions for the GAN model
- post_processing.py
    - Contains plotting code
- EVGen.yml
    - Conda environment file
-GMM.py
    - Contains code to train a GMM model
  

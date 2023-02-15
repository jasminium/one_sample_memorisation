# Installation guide
1. Download and unzip [Celeb-A.zip - Aligned&Cropped Images](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset to ```data/```  
1. Install PyTorch - https://pytorch.org/get-started/locally
2. Install dependencies
```
pip install tqdm scikit-learn pandas seaborn scikit-image 
```
1. Setup attractive/not attractive labels for the Celeb-A dataset.
```
python utils.py
```

# Run Celeb-A memorisation experiment
1. Train Celeb-A models
```
python train_celeb_a.py
```
2. Run the memorisation metrics
```
python eval_celeb_a.py
```

# Rare concepts lead to memorisation experiment

Launch the `toy_memorisation_extra_dim.ipynb` Jupyter notebook.

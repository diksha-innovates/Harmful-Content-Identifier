# Harmful-Content-Identifier

AIM: Develop a model that can identify nude/vulgar/gore/violent images.

-----------------------

DATASETS USED: 

*x1101/nsfw-full, DarkyMan/nsfw-image-classification* for nude and pornographic images

*NeuralShell/Gore-Blood-Dataset-v1.0* for gore images

*rtb1271/human_fight_detection* for Violent images

*uoft-cs/cifar10* (subsampled to 600 images) as a benign "safe" class, so the model has a real
negative class to compare against instead of being forced to always predict a harmful category.

*Alternatively if you have enough computing resources you can use **deepghs/nsfw_detect** to train your model for nudity/pornographic images/vulgarity.*

----------------------

PURPOSE: One major reason is to identify harmful and not safe to work (NSFW) content from websites to avoid harrassment online, or to make web safer for use for general public. 

----------------------

PROJECT STRUCTURE:

```
harmful_content_identifier/   # importable package
    config.py                 # dataset sources, label map, hyperparameters
    data.py                   # dataset loading, labeling, PyTorch Dataset wrapper
    model.py                  # model definition
    train.py                  # training loop
scripts/
    train.py                  # CLI entrypoint: python scripts/train.py --epochs 10
Identify Harmful Content.ipynb  # original Colab notebook, kept for reference
```

Install dependencies with `pip install -r requirements.txt`, then run `python scripts/train.py`.

No images are scraped from the web by this project. All training images are streamed at run
time via `datasets.load_dataset(...)` from the Hugging Face datasets listed above; none are
stored in this repository.

----------------------

METRICS:

Training tracks more than loss/accuracy: per-class precision/recall/F1, macro-F1, and a confusion
matrix are computed on the validation split every epoch.

- `logs/training_log.csv` - per-epoch loss, accuracy, and macro precision/recall/F1
- `metrics/final_validation_metrics.json` - full per-class breakdown for the best checkpoint
- `metrics/confusion_matrix.png` - confusion matrix for the best checkpoint
- `checkpoints/best_model.pt` - weights of the epoch with the highest validation macro-F1

Training also uses a class-weighted loss (the source datasets range from ~190 to tens of
thousands of images), a learning-rate scheduler that reduces LR on a validation macro-F1
plateau, and early stopping once macro-F1 stops improving.

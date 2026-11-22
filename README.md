# Harmful-Content-Identifier

AIM: Develop a model that can identify nude/vulgar/gore/violent images.

-----------------------

DATASETS USED: 

*x1101/nsfw-full, DarkyMan/nsfw-image-classification* for nude and pornographic images

*NeuralShell/Gore-Blood-Dataset-v1.0* for gore images

*rtb1271/human_fight_detection* for Violent images

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
time via `datasets.load_dataset(...)` from the four Hugging Face datasets listed above; none are
stored in this repository.

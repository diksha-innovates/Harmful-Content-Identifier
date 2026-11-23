# Images

This project does **not** scrape images from the web. There is no scraping code anywhere in
this repository - every image used for training is streamed directly from the Hugging Face
Hub at run time via `datasets.load_dataset(...)` (see the notebook / `harmful_content_identifier`
package). Nothing is cached to disk by default and nothing is committed to this repo.

Source datasets:

| Category   | Dataset                                        |
|------------|-------------------------------------------------|
| Nudity/NSFW | [`DarkyMan/nsfw-image-classification`](https://huggingface.co/datasets/DarkyMan/nsfw-image-classification) |
| Nudity/NSFW | [`x1101/nsfw-full`](https://huggingface.co/datasets/x1101/nsfw-full) |
| Gore       | [`NeuralShell/Gore-Blood-Dataset-v1.0`](https://huggingface.co/datasets/NeuralShell/Gore-Blood-Dataset-v1.0) |
| Violence   | [`rtb1271/human_fight_detection`](https://huggingface.co/datasets/rtb1271/human_fight_detection) |

## Why there are no sample images in this folder

This repo intentionally does not commit real NSFW/gore/violent sample images:

- Several of the source datasets are explicitly tagged `not-for-all-audiences` by their authors,
  and their licenses don't clearly grant redistribution rights for a derivative copy.
- Explicit/graphic content sitting in a public git history is hard to remove later and is not
  something to check in casually.
- A git repo is a poor fit for binary image assets anyway (every version is kept forever,
  bloating clone size).

## Pulling local samples for inspection

If you want a handful of local images to sanity-check preprocessing or look at what the model
is training on, run:

```bash
pip install -r requirements.txt
python scripts/download_sample_images.py --per-category 5
```

This downloads a small, configurable number of images per category into `images/samples/`,
which is excluded from version control via `.gitignore` - review the relevant dataset's license
on Hugging Face before using or sharing anything it downloads.

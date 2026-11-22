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

IMAGES: No images are scraped from the web by this project - there is no scraping code in this
repo. Every training image is streamed directly from the Hugging Face datasets listed above via
`datasets.load_dataset(...)`; none are stored here. See `images/README.md` for dataset sources
and how to pull a handful of local samples for inspection.

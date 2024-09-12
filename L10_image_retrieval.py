# Lesson 10: Image Retrieval
# In the classroom, the libraries are already installed for you.
# If you would like to run this code on your own machine, you can install the following:
#     !pip install transformers
#     !pip install torch

# Here is some code that suppresses warning messages.
from transformers.utils import logging
logging.set_verbosity_error()

# Load the model and the processor
from transformers import BlipForImageTextRetrieval

model = BlipForImageTextRetrieval.from_pretrained(
    "./models/Salesforce/blip-itm-base-coco")

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(
    "./models/Salesforce/blip-itm-base-coco")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
 
from PIL import Image
import requests

raw_image =  Image.open(
    requests.get(img_url, stream=True).raw).convert('RGB')

raw_image

# Test, if the image matches the text¶

text = "an image of a woman and a dog on the beach"

inputs = processor(images=raw_image,
                   text=text,
                   return_tensors="pt")

inputs

itm_scores = model(**inputs)[0]

itm_scores

import torch

# Use a softmax layer to get the probabilities

itm_score = torch.nn.functional.softmax(
    itm_scores,dim=1)

itm_score

print(f"""\
The image and text are matched \
with a probability of {itm_score[0][1]:.4f}""")
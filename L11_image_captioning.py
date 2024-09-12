# Lesson 11: Image Captioning
# In the classroom, the libraries are already installed for you.
# If you would like to run this code on your own machine, you can install the following:
#     !pip install transformers
# Here is some code that suppresses warning messages.

from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")

from transformers import BlipForConditionalGeneration

model = BlipForConditionalGeneration.from_pretrained(
    "./models/Salesforce/blip-image-captioning-base")

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(
    "./models/Salesforce/blip-image-captioning-base")

# Load the image.
from PIL import Image

image = Image.open("./beach.jpeg")

image

# Conditional Image Captioning

text = "a photograph of"
inputs = processor(image, text, return_tensors="pt")

inputs

out = model.generate(**inputs)

out

print(processor.decode(out[0], skip_special_tokens=True))

# Unconditional Image Captioning

inputs = processor(image,return_tensors="pt")

out = model.generate(**inputs)

print(processor.decode(out[0], skip_special_tokens=True))


# Lesson 12: Visual Question & Answering
# In the classroom, the libraries are already installed for you.
# If you would like to run this code on your own machine, you can install the following:
#     !pip install transformers
# Here is some code that suppresses warning messages.

from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")

from transformers import BlipForQuestionAnswering

model = BlipForQuestionAnswering.from_pretrained(
    "./models/Salesforce/blip-vqa-base")

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(
    "./models/Salesforce/blip-vqa-base")

from PIL import Image

image = Image.open("./beach.jpeg")

image

question = "how many dogs are in the picture?"

inputs = processor(image, question, return_tensors="pt")

out = model.generate(**inputs)

print(processor.decode(out[0], skip_special_tokens=True))


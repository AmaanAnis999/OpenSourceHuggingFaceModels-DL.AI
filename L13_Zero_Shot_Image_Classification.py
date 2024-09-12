# Lesson 13: Zero-Shot Image Classification
# In the classroom, the libraries are already installed for you.
# If you would like to run this code on your own machine, you can install the following:
#     !pip install transformers
# Load the model and the processor.
# Here is some code that suppresses warning messages.

from transformers.utils import logging
logging.set_verbosity_error()

from transformers import CLIPModel

model = CLIPModel.from_pretrained(
    "./models/openai/clip-vit-large-patch14")

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(
    "./models/openai/clip-vit-large-patch14")

from PIL import Image

image = Image.open("./kittens.jpeg")

image

labels = ["a photo of a cat", "a photo of a dog"]

inputs = processor(text=labels,
                   images=image,
                   return_tensors="pt",
                   padding=True)

outputs = model(**inputs)

outputs

outputs.logits_per_image

probs = outputs.logits_per_image.softmax(dim=1)[0]

probs

probs = list(probs)
for i in range(len(labels)):
  print(f"label: {labels[i]} - probability of {probs[i].item():.4f}")
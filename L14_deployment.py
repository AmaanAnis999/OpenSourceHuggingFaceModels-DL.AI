# (Optional) Lesson 14: Deploy ML Models on 🤗 Hub using Gradio
# Welcome to the last lesson - ML deployment using 🤗 Hub and Gradio libraries.
# This lesson is optional. You can watch the video first to see a walkthrough of how to deploy to Hugging Face Spaces.
# If you would like to follow along or deploy to Hugging Face Spaces later, you can do so by creating a free account on https://huggingface.co/
# You are not required to create an account to complete this lesson, as this lesson contains screenshots and instructions for how to deploy, but does not have any code that requires you to have a Hugging Face account.
# In the classroom, the libraries are already installed for you.
# If you would like to run this code on your own machine, you can install the following:
#     !pip install transformers
#     !pip install gradio
#     !pip install gradio_client
# Note that if you run into issues when making an API call to your own space, you can try to upgrade your version of gradio_client:
# pip install -U gradio_client
# Here is some code that suppresses warning messages.

from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore", 
                        message="Using the model-agnostic default `max_length`")

#  Spaces
# You can create an account on hugging face from here, to follow the instructions provided in the video.
# App Demo: Image Captioning
# Load the model and create an app interface using Gradio to perform Image Captioning.
# Troubleshooting Tip
# Note, in the classroom, you may see the code for creating the Gradio app run indefinitely.
# This is specific to this classroom environment when it's serving many learners at once, and you won't wouldn't experience this issue if you run this code on your own machine.
# To fix this, please restart the kernel (Menu Kernel->Restart Kernel) and re-run the code in the lab from the beginning of the lesson.

import os
import gradio as gr
from transformers import pipeline

pipe = pipeline("image-to-text",
                model="./models/Salesforce/blip-image-captioning-base")

def launch(input):
    out = pipe(input)
    return out[0]['generated_text']

iface = gr.Interface(launch,
                     inputs=gr.Image(type='pil'),
                     outputs="text")

iface.launch(share=True, 
             server_port=int(os.environ['PORT1']))

iface.close()


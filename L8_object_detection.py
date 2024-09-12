# !pip install transformers
#     !pip install gradio
#     !pip install timm
#     !pip install inflect
#     !pip install phonemizer

from helper import load_image_from_url, render_results_in_image

from transformers import pipeline

from transformers.utils import logging
logging.set_verbosity_error()

from helper import ignore_warnings
ignore_warnings()

od_pipe = pipeline("object-detection", "facebook/detr-resnet-50")

from PIL import Image

raw_image = Image.open('huggingface_friends.jpg')
raw_image.resize((569, 491))

pipeline_output = od_pipe(raw_image)

processed_image = render_results_in_image(
    raw_image, 
    pipeline_output)

processed_image

# Using Gradio as a Simple Interface
# Use Gradio to create a demo for the object detection app.
# The demo makes it look friendly and easy to use.
# You can share the demo with your friends and colleagues as well.

import os
import gradio as gr

def get_pipeline_prediction(pil_image):
    
    pipeline_output = od_pipe(pil_image)
    
    processed_image = render_results_in_image(pil_image,
                                            pipeline_output)
    return processed_image

demo = gr.Interface(
  fn=get_pipeline_prediction,
  inputs=gr.Image(label="Input image", 
                  type="pil"),
  outputs=gr.Image(label="Output image with predicted instances",
                   type="pil")
)

demo.launch(share=True, server_port=int(os.environ['PORT1']))

demo.close()

# Make an AI Powered Audio Assistant
# Combine the object detector with a text-to-speech model that will help dictate what is inside the image.

# Inspect the output of the object detection pipeline.

pipeline_output

od_pipe

raw_image = Image.open('huggingface_friends.jpg')
raw_image.resize((284, 245))

from helper import summarize_predictions_natural_language

text = summarize_predictions_natural_language(pipeline_output)

text

tts_pipe = pipeline("text-to-speech",
                    model="./models/kakao-enterprise/vits-ljs")

narrated_text = tts_pipe(text)


### Play the Generated Audio
from IPython.display import Audio as IPythonAudio

IPythonAudio(narrated_text["audio"][0],
             rate=narrated_text["sampling_rate"])
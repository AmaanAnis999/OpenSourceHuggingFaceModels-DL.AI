# !pip install transformers
#   !pip install datasets
#   !pip install soundfile
#   !pip install librosa

from transformers.utils import logging
logging.set_verbosity_error()

from datasets import load_dataset, load_from_disk

# This dataset is a collection of different sounds of 5 seconds
# dataset = load_dataset("ashraq/esc50",
#                       split="train[0:10]")
dataset = load_from_disk("./models/ashraq/esc50/train")

audio_sample = dataset[0]

audio_sample

from IPython.display import Audio as IPythonAudio
IPythonAudio(audio_sample["audio"]["array"],
             rate=audio_sample["audio"]["sampling_rate"])

from transformers import pipeline

zero_shot_classifier = pipeline(
    task="zero-shot-audio-classification",
    model="./models/laion/clap-htsat-unfused")

(1 * 192000) / 16000
(5 * 192000) / 16000

zero_shot_classifier.feature_extractor.sampling_rate

audio_sample["audio"]["sampling_rate"]

from datasets import Audio

dataset = dataset.cast_column(
    "audio",
     Audio(sampling_rate=48_000))

audio_sample = dataset[0]

audio_sample

candidate_labels = ["Sound of a dog",
                    "Sound of vacuum cleaner"]

zero_shot_classifier(audio_sample["audio"]["array"],
                     candidate_labels=candidate_labels)

candidate_labels = ["Sound of a child crying",
                    "Sound of vacuum cleaner",
                    "Sound of a bird singing",
                    "Sound of an airplane"]

zero_shot_classifier(audio_sample["audio"]["array"],
                     candidate_labels=candidate_labels)


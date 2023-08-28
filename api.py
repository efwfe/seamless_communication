import os
from fastapi import FastAPI
import gradio as gr
import numpy as np
import torch
import torchaudio
from seamless_communication.models.inference.translator import Translator
import uvicorn
from fastapi.responses import JSONResponse
import json

app = FastAPI()
AUDIO_SAMPLE_RATE = 16000.0
MAX_INPUT_AUDIO_LENGTH = 60  # in seconds
DEFAULT_TARGET_LANGUAGE = "English"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
translator = Translator(
    model_name_or_card="seamlessM4T_large",
    vocoder_name_or_card="vocoder_36langs",
    device=device,
    dtype=torch.float32
)

from lang_list import (
    LANGUAGE_NAME_TO_CODE,
    S2ST_TARGET_LANGUAGE_NAMES,
    S2TT_TARGET_LANGUAGE_NAMES,
    T2TT_TARGET_LANGUAGE_NAMES,
    TEXT_SOURCE_LANGUAGE_NAMES,
)

def predict(
    task_name: str,
    audio_source: str,
    input_audio_mic: str | None,
    input_audio_file: str | None,
    input_text: str | None,
    source_language: str | None,
    target_language: str):
    task_name = task_name.split()[0]
    source_language_code = LANGUAGE_NAME_TO_CODE[source_language] if source_language else None
    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]

    if task_name in ["S2ST", "S2TT", "ASR"]:
        if audio_source == "microphone":
            input_data = input_audio_mic
        else:
            input_data = input_audio_file

        arr, org_sr = torchaudio.load(input_data)
        new_arr = torchaudio.functional.resample(arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE)
        max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
        if new_arr.shape[1] > max_length:
            new_arr = new_arr[:, :max_length]
            gr.Warning(f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used.")
        torchaudio.save(input_data, new_arr, sample_rate=int(AUDIO_SAMPLE_RATE))
    else:
        input_data = input_text
    text_out, wav, sr = translator.predict(
        input=input_data,
        task_str=task_name,
        tgt_lang=target_language_code,
        src_lang=source_language_code,
        ngram_filtering=True,
    )
    if task_name in ["S2ST", "T2ST"]:
        return (sr, wav.cpu().detach().numpy()), text_out
    else:
        return None, text_out


def process_t2tt_example(
    input_text: str, source_language: str, target_language: str
):
    return predict(
        task_name="T2TT",
        audio_source="",
        input_audio_mic=None,
        input_audio_file=None,
        input_text=input_text,
        source_language=source_language,
        target_language=target_language,
    )


@app.post("/t2t")
def translate_t2t(text: str, source_lang: str, target_lang: str):
    _, data = process_t2tt_example(text, source_language=source_lang, target_language=target_lang)
    return json.dumps(dict(content=str(data)), default=str)

if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=20000, reload=False)
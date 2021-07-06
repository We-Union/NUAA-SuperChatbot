from ASR.constant import *
import os

def bitstream_to_text(bitstream : bytes) -> str:
    text = client.asr(bitstream, format="wav", rate=16000,  options={'dev_pid': 1537})
    return text

def file_to_text(path : str) -> str:
    with open(path, "rb") as f:
        bitstream = f.read()
    return bitstream_to_text(bitstream)

from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import json
from codebleu import calc_codebleu

model_type = "4bit" # 4bit 8bit org dynamic
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if model_type == "dynamic":
    device = "cpu"
def loadData(filePath):
    with open(filePath, 'r') as f:
        jsondata = json.load(f)

    # data = jsondata['member']
    data = jsondata['nonmember']
    # data.extend(jsondata['nonmember'])
    return data

filePath = "raw_data.json"
data = loadData(filePath=filePath)
# Load model and tokenizer
# // 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B
model_name = "bigcode/starcoder2-3b"

def loadModel(model_name, type):
    if type =="4bit":
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map=device, quantization_config=bnb_config)
    elif type == "8bit":
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_8bit=True
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map=device, quantization_config=bnb_config)
    elif type == "org":
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
    elif type == "dynamic":
        model_copy = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
        model = torch.quantization.quantize_dynamic(
            model_copy, {torch.nn.Linear}, dtype=torch.qint8)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
    return model

model = loadModel(model_name=model_name, type=model_type)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(len(data))
model.eval()

torch.save(model.state_dict(), "temp.p")
print("Model: ", model_name, " | Type: ", model_type, ' | Size (MB):', os.path.getsize("temp.p")/1e6)
os.remove('temp.p')

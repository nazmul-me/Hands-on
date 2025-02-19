from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import codebleu
from codebleu import calc_codebleu

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loadData(filePath):
    with open(filePath, 'r') as f:
        jsondata = json.load(f)

    data = jsondata['member']
    data.extend(jsondata['nonmember'])
    return data

filePath = "raw_data.json"
data = loadData(filePath=filePath)
# Load model and tokenizer
model_name = "EleutherAI/gpt-neo-125m"

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
        print("Dynamic quantization complete\n")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
    return model

model = loadModel(model_name=model_name, type='org') # 4bit 8bit org dynamic
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(len(data))

res = []
for d in data[0:1900]:
    prompt = d[:200]
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(**inputs, num_return_sequences=1, max_length=1024)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(d)
    print("---------------------------------------")
    print(generated_code)

    result = calc_codebleu([d], [generated_code], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=tokenizer)
    res.append(result)
    break
print(len(res))
averages = {key: 0 for key in res[0].keys()}
for entry in res:
    for key, value in entry.items():
        averages[key] += value

averages = {key: value / len(res) for key, value in averages.items()}

print("Averages:")
for key, value in averages.items():
    print(f"{key}: {value}")

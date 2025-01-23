from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model_name = "EleutherAI/gpt-neo-125m"

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
# bnb_config = transformers.BitsAndBytesConfig(
#     load_in_8bit=True
# )
model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map=device, quantization_config=bnb_config)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)

tokenizer = AutoTokenizer.from_pretrained(model_name)


import json

with open("raw_data.json", 'r') as f:
    jsondata = json.load(f)

data = jsondata['member']
data.extend(jsondata['nonmember'])
print("sample size", len(data))

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(data[0:3], return_tensors="pt", padding=True).to(device)

outputs = model(**inputs, labels=inputs["input_ids"])
# Logits returned by the model
logits = outputs.logits

# Get predicted tokens
predicted_token_ids = torch.argmax(logits, dim=-1)
print(predicted_token_ids.shape)
print(predicted_token_ids[0].shape)
# Decode the predicted tokens into code
generated_code = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
print(len(generated_code))
pred_tokens = predicted_token_ids.detach().cpu().numpy()

# np.savetxt('pred_tokens.txt', pred_tokens, delimiter=',', fmt="%s")
# with open("gen_tokens.txt", "w") as file:
#     # Write the string to the file
#     file.write(generated_code)


from codebleu import calc_codebleu
prediction = generated_code
reference = data[0:3]
print(type(prediction), type(reference))
result = calc_codebleu([reference], [prediction], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
print(result)
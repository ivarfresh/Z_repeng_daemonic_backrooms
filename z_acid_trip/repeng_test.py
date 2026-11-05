import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from repeng import ControlVector, ControlModel, DatasetEntry

# First we import the libraries, including the repeng library I wrote, 
# load Mistral-7B, and wrap it in a ControlModel for later.
model_name = "microsoft/DialoGPT-medium"
print(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = 0
print("Tokenizer loaded successfully")

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Moving model to device: {device}")

model = model.to(device)
model = ControlModel(model, list(range(-5, -18, -1)))
print("ControlModel created with layers:", list(range(-5, -18, -1)))

user_tag, asst_tag = "", ""  # DialoGPT doesn't use special tags

#Next we generate the honest v.s. untruthful dataset,
# using a JSON file true_facts.json (taken from the paper authors) of about 
# ~300 simple facts to use as suffixes, which is augmented by truncation. 
# This generation step runs essentially instantly.

# with open("data/true_facts.json") as f:
#     suffixes = json.load(f)

# # the control vector we're going to make is honest / untruthful, like the paper
# positive_personas = ["honest"]
# negative_personas = ["untruthful"]
# def template(persona: str, suffix: str) -> str:
#     return f"{user_tag} Pretend you're an {persona} person making statements about the world. {asst_tag} {suffix}"

# dataset = []
# for suffix in suffixes:
#     tokens = tokenizer.tokenize(suffix)
#     # we augment our short suffix list by taking lots of different truncations.
#     # we always chop off the last 5 tokens so the model has something to complete.
#     for i in range(1, len(tokens) - 5):
#         truncated = tokenizer.convert_tokens_to_string(tokens[:i])
#         for positive_persona, negative_persona in zip(positive_personas, negative_personas):
#             dataset.append(
#                 DatasetEntry(
#                     positive=template(positive_persona, truncated),
#                     negative=template(negative_persona, truncated),
#                 )
#             )
with open("data/true_facts.json") as f:
    suffixes = json.load(f)

print(f"Loaded {len(suffixes)} suffixes from true_facts.json")
print(f"First few suffixes: {suffixes[:3]}")

# the control vector we're going to make is honest / untruthful, like the paper
positive_personas = ["honest"]
negative_personas = ["untruthful"]
def template(persona: str, suffix: str) -> str:
    return f"{user_tag} Pretend you're an {persona} person making statements about the world. {asst_tag} {suffix}"

dataset = []
print(f"Starting dataset creation with {len(suffixes)} suffixes...")

for i, suffix in enumerate(suffixes):
    print(f"\nProcessing suffix {i+1}/{len(suffixes)}: {suffix[:50]}...")
    
    tokens = tokenizer.tokenize(suffix)
    print(f"  Tokenized into {len(tokens)} tokens")
    
    # we augment our short suffix list by taking lots of different truncations.
    # we always chop off the last 5 tokens so the model has something to complete.
    truncation_range = range(1, len(tokens) - 5)
    print(f"  Will create {len(truncation_range)} truncated versions")
    
    for j, i in enumerate(truncation_range):
        if j < 3:  # Only print first few for brevity
            print(f"    Truncation {j+1}: keeping first {i} tokens")
        
        truncated = tokenizer.convert_tokens_to_string(tokens[:i])
        
        for positive_persona, negative_persona in zip(positive_personas, negative_personas):
            positive_template = template(positive_persona, truncated)
            negative_template = template(negative_persona, truncated)
            
            if j < 2:  # Only print first few examples
                print(f"      Positive: {positive_template[:100]}...")
                print(f"      Negative: {negative_template[:100]}...")
            
            dataset.append(
                DatasetEntry(
                    positive=positive_template,
                    negative=negative_template,
                )
            )

print(f"\nDataset creation complete! Total entries: {len(dataset)}")
print(f"First dataset entry:")
print(f"  Positive: {dataset[0].positive}")
print(f"  Negative: {dataset[0].negative}")
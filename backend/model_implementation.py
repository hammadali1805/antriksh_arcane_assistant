import pandas as pd
import numpy as np
import os

def read_txt(file_path):
    text = ""
    try:
        with open(file_path, "r") as file:
            text = file.read()
    except:
        text = ""
    return text

with open("train.txt", "w") as f:
    f.write('')
    
data = ""
for filename in os.listdir("./"):
    file_path = os.path.join("./", filename)
    if file_path.endswith(".txt") and (file_path != "train.txt" or file_path != "requirements.txt"):
        data += read_txt(file_path)
        data =  ' '.join(data.split('\n'))
            
        with open("train.txt", "a") as f:
            f.write(data)
            
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments

def load_dataset(file_path, tokenizer, block_size = 128):
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = block_size,
    )
    return dataset

def load_data_collator(tokenizer, mlm = False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=mlm,
    )
    return data_collator

def train(train_file_path,model_name,
        output_dir,
        overwrite_output_dir,
        per_device_train_batch_size,
        num_train_epochs,
        save_steps):
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    train_dataset = load_dataset(train_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)
    tokenizer.save_pretrained(output_dir)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.save_pretrained(output_dir)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
      
    trainer.train()
    trainer.save_model()

train_file_path = "train.txt"
model_name = 'gpt2'
output_dir = './custom_model'
overwrite_output_dir = False
per_device_train_batch_size = 8
num_train_epochs = 50.0
save_steps = 50000

train(
    train_file_path=train_file_path,
    model_name=model_name,
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps
)

from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer

def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def generate_text(model_path, sequence, max_length):
    
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))

model_path = "./custom_model"
sequence = "" #Enter your prompt here
max_len = 50
generate_text(model_path, sequence, max_len)
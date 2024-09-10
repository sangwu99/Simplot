import argparse

import torch
import torch.nn as nn

import pandas as pd
import os

from torch.utils.data import Dataset, DataLoader
from PIL import Image

import gc

from PIL import Image
import numpy as np
import json

from tqdm import tqdm
from collections import defaultdict
from data import SimplotTestDataset
from transformers import DonutProcessor, VisionEncoderDecoderModel
import copy
    
def get_table(csv):
    new_label = ''
    for idx in range(len(csv)):
        values = csv.values[idx]
        if idx==0:
            for i in range(len(values)):
                if i==0:
                    new_label += values[i]
                else:
                    new_label += f' | {values[i]}'
        else:
            for i in range(len(values)):
                if i==0:
                    new_label += f' <0x0A> {values[i]}'
                else:
                    new_label += f' | {values[i]}'
    return new_label

def get_is_nan(text):
    try:
        return float(text)
    except:
        return np.nan

            
def rlstrip_4_column(csv):
    if len(csv.columns) > 1:
        text_list = []
        for idx in range(len(csv.columns)):
            text = str(csv[idx][0])
            text = text.rstrip()
            text = text.lstrip()
            text_list.append(text)
        return " | ".join(text_list)
    else:
        text = str(csv[0][0])
        text = text.rstrip()
        text = text.lstrip()
        return text
    
def rlstrip_4_index(csv):
    text_list = []
    for idx in range(len(csv.index)):
        text = str(csv[0][idx])
        text = text.rstrip()
        text = text.lstrip()
        text_list.append(text)
    return " | ".join(text_list)


def collator(batch):
  new_batch = {"flattened_patches":[], 
               "attention_mask":[]}
  texts = [item["text"] for item in batch]
  
  text_inputs = processor.tokenizer(text=texts, padding="max_length", return_tensors="pt", add_special_tokens=True, max_length=800)

  new_batch["labels"] = text_inputs.input_ids
  
  for item in batch:
    new_batch["flattened_patches"].append(item["flattened_patches"])
    new_batch["attention_mask"].append(item["attention_mask"])
  
  new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
  new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

  return new_batch

def get_lrstrip(text):
    text = str(text)
    text = text.rstrip()
    text = text.lstrip()
    text = text.rstrip('%')
    if text.endswith(".0"):
        text = text[:-2]
    
    return text

def get_table_uni(csv):
    new_label = ''
    for idx in range(len(csv)):
        values = csv.values[idx]
        if idx==0:
            for i in range(len(values)):
                if i==0:
                    new_label += values[i]
                else:
                    new_label += f' | {values[i]}'
        else:
            for i in range(len(values)):
                if i==0:
                    new_label += f' & {values[i]}'
                else:
                    new_label += f' | {values[i]}'
    return new_label



def main(args):
    
    torch.cuda.empty_cache()
    gc.collect()

    processor = DonutProcessor.from_pretrained(args.checkpoint_path)
    model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint_path)
    
    processor.tokenizer.add_tokens('<columns>')
    processor.tokenizer.add_tokens('<rows>')
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
        
    model.teacher_encoder = copy.deepcopy(model.encoder)
    model.load_state_dict(torch.load(args.model_state_dict, map_location='cpu'))
    
    dataset = []
  
    img_path = args.img_path
    table_path = args.table_path
    row_path = args.row_path
    col_path = args.col_path
    json_path = args.json_path
    
    img_list = sorted(os.listdir(img_path))
    failed_list = []

    for img in tqdm(img_list):
        try: 
            with open(os.path.join(json_path, img[:-3] + 'json')) as f:
                data_json = json.load(f)
            
                text = pd.read_csv(os.path.join(table_path, img[:-3]+'csv'), header = None)
                text = get_table_uni(text)
                
                row = pd.read_csv(os.path.join(row_path, img[:-3]+'csv'))
                row.columns = [0]
                row = rlstrip_4_index(row)
                
                col = pd.read_csv(os.path.join(col_path, img[:-3]+'csv'))
                col.columns = [0]
                col = rlstrip_4_index(col)
                
                dataset.append({'image': os.path.join(img_path,img), 
                                'text' : text,
                                'row' : row,
                                'col' : col,
                                'img_name' : img})
        except:
            failed_list.append({'failed_csv':img[:-3]+'csv'})

    
    test_dataset = SimplotTestDataset(dataset, processor = processor)
    
    gpu = args.device
    
    print(gpu)
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    
    print(device)
    
    model.to(device)
    model.eval()
    
    accuracy_list = {'img':[],'type':[],'pred':[],'label':[]}
        
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)
    
    for idx, batch in enumerate(test_dataloader):
        input_tensor, input_ids, label = batch
        
        output = model.generate(input_tensor.to(device),
                                decoder_input_ids = input_ids.to(device),
                                max_length = model.decoder.config.max_position_embeddings,
                                early_stopping = True,
                                pad_token_id = processor.tokenizer.pad_token_id,
                                eos_token_id = processor.tokenizer.eos_token_id,
                                use_cache = True,
                                num_beams = 4,
                                bad_words_ids = [[processor.tokenizer.unk_token_id]],
                                return_dict_in_generate=True)
        
        pred = processor.batch_decode(output.sequences)[0]
        pred = pred.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        pred = pred.split("<s_answer>")[1].strip()
        print(pred)
        print(label[0])
        
        accuracy_list['img'].append(dataset[idx]['image'])
        accuracy_list['pred'].append(pred)
        accuracy_list['label'].append(label[0])
            
    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    
    result_df = pd.DataFrame(accuracy_list)
    result_df.to_csv(os.path.join(result_path, 'prediction.csv'))
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    
    parser.add_argument('--img_path', type=str, default='../data/test/png')
    parser.add_argument('--table_path', type=str, default='../data/test/tables')
    parser.add_argument('--row_path', type=str, default='../data/test/gpt_indexes')
    parser.add_argument('--col_path', type=str, default='../data/test/gpt_columns')
    parser.add_argument('--json_path', type=str, default='../data/test/annotations')    
    parser.add_argument('--device', type=str, default='6')

    parser.add_argument('--result_path', type=str, default='../result/unichart/')
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--theta', type=float, default=0.5)
    
    parser.add_argument('--checkpoint_path', type=str, default="ahmed-masry/unichart-base-960")
    parser.add_argument('--model_state_dict', type=str, default='../state/unichart/phase2/phase_2_best_model.pth')
    
    args = parser.parse_args()
    
    main(args)
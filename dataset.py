import os 
import json
import torch
import pandas as pd 

from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

from utils.utils import get_table, rlstrip_4_index, rlstrip_4_column


class SimplotDataset(Dataset):
    def __init__(self, dataset, processor, phase):
        self.dataset = dataset
        self.processor = processor
        self.phase = phase

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        num_rows = len(str(item['row']).split('|'))
        num_cols = len(str(item['col']).split('|'))
        text = f"generate underlying data table of the figure below given the columns ({item['col']}); and the rows ({item['row']}) \n a number of columns are {num_cols} and a number of rows are {num_rows}"
        
        encoding = {}
        encoding['image'] = item['image']
        encoding['render'] = text
        encoding['processor'] = self.processor
        encoding['text'] = item['text']

        if self.phase == 1:
            return encoding
        elif self.phase == 2:
            encoding['positive_image'] = item['positive_image']
            encoding['negative_image'] = item['negative_image']
            return encoding
        else:
            encoding['type'] = item['type']
            return encoding
        
def phase_1_collator(batch):
    new_batch = {"flattened_patches":[], 
            "attention_mask":[]}
    processor = batch[0]["processor"]
    texts = [item["text"] for item in batch]
    encodings = []
    for item in batch:
        images = Image.open(item["image"]).convert("RGB")
        render = item["render"]
        encoding = processor(images=images, text = render , return_tensors="pt", add_special_tokens=True, max_patches=1024)
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encodings.append(encoding)

    text_inputs = processor.tokenizer(text=texts, padding="max_length", return_tensors="pt", add_special_tokens=True, max_length=800, truncation=True)

    new_batch["labels"] = text_inputs.input_ids
    
    for item in encodings:
        new_batch["flattened_patches"].append(item["flattened_patches"])
        new_batch["attention_mask"].append(item["attention_mask"])
    
    new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

    return new_batch

def phase_2_collator(batch):
    new_batch = {"flattened_patches":[], 
               "attention_mask":[],}
    processor = batch[0]["processor"]
    texts = [item["text"] for item in batch]
    encodings = []
    positive_encodings = []
    negative_encodings = []
    
    for item in batch:
        images = Image.open(item["image"]).convert("RGB")
        positive_images = Image.open(item["positive_image"]).convert("RGB")
        negative_images = Image.open(item["negative_image"]).convert("RGB")
        render = item['render']

        encoding = processor(images=images, text = render , return_tensors="pt", add_special_tokens=True, max_patches=1024)
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        positive_encoding = processor(images=positive_images, text = render , return_tensors="pt", add_special_tokens=True, max_patches=1024)
        positive_encoding = {k:v.squeeze() for k,v in positive_encoding.items()}
        negative_encoding = processor(images=negative_images, text = render , return_tensors="pt", add_special_tokens=True, max_patches=1024)
        negative_encoding = {k:v.squeeze() for k,v in negative_encoding.items()}

        encodings.append(encoding)
        positive_encodings.append(positive_encoding)
        negative_encodings.append(negative_encoding)
  
    text_inputs = processor.tokenizer(text=texts, padding="max_length", return_tensors="pt", add_special_tokens=True, max_length=800, truncation=True)

    new_batch["labels"] = text_inputs.input_ids
  
    positive_flattened_patches = []
    positive_attention_mask = []
    
    negative_flattened_patches = []
    negative_attention_mask = []
  
    for item in encodings:
        new_batch["flattened_patches"].append(item["flattened_patches"])
        new_batch["attention_mask"].append(item["attention_mask"])
        
    for item in positive_encodings:
        positive_flattened_patches.append(item["flattened_patches"])
        positive_attention_mask.append(item["attention_mask"])
        
    for item in negative_encodings:
        negative_flattened_patches.append(item["flattened_patches"])
        negative_attention_mask.append(item["attention_mask"])

    new_batch["flattened_patches"].extend(positive_flattened_patches)
    new_batch["attention_mask"].extend(positive_attention_mask)
    new_batch["flattened_patches"].extend(negative_flattened_patches)
    new_batch["attention_mask"].extend(negative_attention_mask)

    new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

    return new_batch

def test_collator(batch):
    new_batch = {"flattened_patches":[], 
            "attention_mask":[],}
    processor = batch[0]["processor"]
    texts = [item["text"] for item in batch]
    encodings = []
    for item in batch:
        images = Image.open(item["image"]).convert("RGB")
        render = item["render"]
        encoding = processor(images=images, text = render , return_tensors="pt", add_special_tokens=True, max_patches=1024)
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encodings.append(encoding)

    text_inputs = processor.tokenizer(text=texts, padding="max_length", return_tensors="pt", add_special_tokens=True, max_length=800, truncation=True)

    new_batch["labels"] = text_inputs.input_ids
    
    for item in encodings:
        new_batch["flattened_patches"].append(item["flattened_patches"])
        new_batch["attention_mask"].append(item["attention_mask"])
    
    new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])
    new_batch['type'] = [item['type'] for item in batch]

    return new_batch

def prepare_dataset(args):
    img_path = args.img_path
    table_path = args.table_path
    row_path = args.row_path
    col_path = args.col_path
    pos_img_path = args.pos_img_path
            
    test_img_path = args.test_img_path
    test_table_path = args.test_table_path
    test_row_path = args.test_row_path
    test_col_path = args.test_col_path
    pos_test_img_path = args.pos_test_img_path
    
    img_list = sorted(os.listdir(img_path))
    pos_img_list = sorted(os.listdir(pos_img_path))

    test_img_list = sorted(os.listdir(test_img_path))
    pos_test_img_list = sorted(os.listdir(pos_test_img_path))
    
    dataset = []
    test_dataset = []
    
    if args.phase == 1:
        for img in tqdm(pos_img_list):
            try: 
                text = pd.read_csv(os.path.join(table_path, img[:-3]+'csv'), header = None)
                text = get_table(text)
                
                row = pd.read_csv(os.path.join(row_path, img[:-3]+'csv'), header = None)
                row = rlstrip_4_index(row)
                
                col = pd.read_csv(os.path.join(col_path, img[:-3]+'csv'), header = None)
                col = rlstrip_4_column(col)
                
                dataset.append({'image': os.path.join(pos_img_path,img), 
                                'text' : text,
                                'row' : row,
                                'col' : col,
                                'img_path' : img,
                                })
                
            except:
                pass
            
        for img in tqdm(pos_test_img_list):
            try:
                text = pd.read_csv(os.path.join(test_table_path, img[:-3]+'csv'), header = None)
                text = get_table(text)
                
                row = pd.read_csv(os.path.join(test_row_path, img[:-3]+'csv'), header = None)
                row = rlstrip_4_index(row)
                
                col = pd.read_csv(os.path.join(test_col_path, img[:-3]+'csv'), header = None)
                col = rlstrip_4_column(col)
                
                test_dataset.append({'image': os.path.join(pos_test_img_path,img), 
                                    'text' : text,
                                    'row' : row,
                                    'col' : col,
                                    'img_path' : img})
            except:
                pass

            
        return dataset, test_dataset

    else:
        neg_img_path = args.neg_img_path
        neg_img_list = sorted(os.listdir(neg_img_path))
        
        for img in tqdm(img_list):
            try: 
                text = pd.read_csv(os.path.join(table_path, img[:-3]+'csv'), header = None)
                text = get_table(text)
                
                row = pd.read_csv(os.path.join(row_path, img[:-3]+'csv'), header = None)
                row = rlstrip_4_index(row)
                
                col = pd.read_csv(os.path.join(col_path, img[:-3]+'csv'), header = None)
                col = rlstrip_4_column(col)
                
                if (img not in pos_img_list) or (img not in neg_img_list):
                    continue
                
                dataset.append({'image': os.path.join(img_path,img), 
                                'positive_image' : os.path.join(pos_img_path, img),
                                'negative_image' : os.path.join(neg_img_path, img),
                                'text' : text,
                                'row' : row,
                                'col' : col,
                                'img_path' : img,
                                })
                
            except:
                pass

        for img in tqdm(test_img_list):
            try:
                text = pd.read_csv(os.path.join(test_table_path, img[:-3]+'csv'), header = None)
                text = get_table(text)
                
                row = pd.read_csv(os.path.join(test_row_path, img[:-3]+'csv'), header = None)
                row = rlstrip_4_index(row)
                
                col = pd.read_csv(os.path.join(test_col_path, img[:-3]+'csv'), header = None)
                col = rlstrip_4_column(col)
                
                test_dataset.append({'image': os.path.join(test_img_path,img), 
                                    'text' : text,
                                    'row' : row,
                                    'col' : col,
                                    'img_path' : img})
            except:
                pass

        return dataset, test_dataset
    

def prepare_test_dataset(args):
    dataset = []
    
    img_path = args.img_path
    table_path = args.table_path
    row_path = args.row_path
    col_path = args.col_path
    json_path = args.json_path
    
    img_list = sorted(os.listdir(img_path))
    for img in tqdm(img_list):
        try: 
            with open(os.path.join(json_path, img[:-3] + 'json')) as f:
                data_json = json.load(f)
            
                text = pd.read_csv(os.path.join(table_path, img[:-3]+'csv'), header = None)
                text = get_table(text)
                
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
                                'type' : data_json['type'],
                                'img_name' : img})
        except:
            pass

    
    return dataset
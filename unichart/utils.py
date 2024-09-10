import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import json

import torch
import pandas as pd 

from PIL import Image
from tqdm import tqdm

def get_lrstrip(text):
    text = str(text)
    text = text.rstrip()
    text = text.lstrip()
    text = text.rstrip('%')
    if text.endswith(".0"):
        text = text[:-2]
    
    return text

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

def get_is_nan(text):
    try:
        return float(text)
    except:
        return np.nan
    

def extract_column_row(table, column_path, index_path, table_path):
    gt = pd.read_csv(os.path.join(table_path,table))
    columns = gt.columns[1:]
    indexes = gt.iloc[:,0]
    pd.DataFrame(columns.to_list()).T.to_csv(path_or_buf=os.path.join(column_path,table),header=False, index=False)
    indexes.to_csv(path_or_buf=os.path.join(index_path,table),header=False, index=False)
    


def modify_string(input_string, max_length=16):
    words = input_string.split()
    
    modified_lines = []
    current_line = words[0]
    
    for word in words[1:]:
        if len(current_line) + len(word) <= max_length:
            current_line += ' ' + word
        else:
            modified_lines.append(current_line.strip())
            current_line = word
    
    if current_line:
        modified_lines.append(current_line.strip())
    
    modified_result = '\n'.join(modified_lines)
    
    return modified_result


def get_strip(list, max_length=16):
    new_list = [] 
    for i in list:
        i = str(i)
        if len(i) > max_length:
            i = modify_string(i, max_length)
        new_list.append(i)
    return new_list

def replace_bad_word(x):
    try:
        return float(x.replace('%',''))
    except:
        return x

def custom_floor(number, decimal_places):
    factor = 10 ** decimal_places
    rounded_number = round(number * factor) / factor
    return rounded_number


def before_annotate(x):
    x = custom_floor(x, 2)
    x = str(x)
    if x.endswith('.0'):
        x = x[:-2]
    return x

def get_max_length(list):
    max_length = 0
    for i in list:
        i = str(i)
        if len(i) > max_length:
            max_length = len(i)
    return max_length

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
    failed_list = []
    
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

    elif args.phase == 2:
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
    
    elif args.phase == 4:
        for img in tqdm(pos_img_list):
            try: 
                text = pd.read_csv(os.path.join(table_path, img[:-3]+'csv'), header = None)
                text = get_table_uni(text)
                
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
                text = get_table_uni(text)
                
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

    elif args.phase == 5:
        neg_img_path = args.neg_img_path
        neg_img_list = sorted(os.listdir(neg_img_path))
        
        for img in tqdm(img_list):
            try: 
                text = pd.read_csv(os.path.join(table_path, img[:-3]+'csv'), header = None)
                text = get_table_uni(text)
                
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
                text = get_table_uni(text)
                
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
            
    else:
        for img in tqdm(img_list):
            try: 
                text = pd.read_csv(os.path.join(table_path, img[:-3]+'csv'), header = None)
                text = get_table(text)
                
                row = pd.read_csv(os.path.join(row_path, img[:-3]+'csv'), header = None)
                row = rlstrip_4_index(row)
                
                col = pd.read_csv(os.path.join(col_path, img[:-3]+'csv'), header = None)
                col = rlstrip_4_column(col)
                
                dataset.append({'image': os.path.join(img_path,img), 
                                'text' : text,
                                'row' : row,
                                'col' : col,
                                'img_path' : img,
                                })
                
            except:
                failed_list.append(img)
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
                failed_list.append(img)
                pass
        print(f'failed length: {len(failed_list)}')

            
        return dataset, test_dataset

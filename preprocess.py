import os
import ast
import json
import argparse
from tqdm import tqdm

from utils.utils import extract_column_row, generate_simple_chart


def main(args):
    
    tasks = ast.literal_eval(args.tasks)
    
    for task in tqdm(tasks):
        table_base_path = f'./data/{task}/tables'
        table_list = os.listdir(table_base_path)
        json_path = f'./data/{task}/annotations/'
        
        index_path = f'./data/{task}/indexes'
        column_path = f'./data/{task}/columns'
        
        positive_path = f'./data/{task}/positive_png'
        negative_path = f'./data/{task}/negative_png'
        
        os.makedirs(index_path, exist_ok=True)
        os.makedirs(column_path, exist_ok=True)
        
        os.makedirs(positive_path, exist_ok=True)
        os.makedirs(negative_path, exist_ok=True)
        
        for table in tqdm(table_list):
            try:
                extract_column_row(table, column_path, index_path, table_base_path)
            except:
                pass 
            try:
                generate_simple_chart(table, table_base_path, json_path, positive_path, negative_path)
            except:
                pass
            
        
        
    print('Finished extracting columns and indexes for all tasks')
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--tasks', type=str, default = '["train","val"]')
    
    args = parser.parse_args()
    main(args)
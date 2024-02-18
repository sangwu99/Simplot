import os
import argparse  
import numpy as np
import pandas as pd 

import torch
from torch.utils.data import DataLoader
from transformers import Pix2StructProcessor

from model import Simplot
from trainer import inference
from dataset import prepare_test_dataset, SimplotDataset, test_collator

def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    processor = Pix2StructProcessor.from_pretrained('google/deplot')
    processor.image_processor.is_vqa = True
    
    np.random.seed(args.seed)
    result_path = args.result_path
    os.makedirs(result_path, exist_ok = True)
    gpu = args.device 
    
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    model = Simplot(args)

    dataset = prepare_test_dataset(args)
    test_dataset = SimplotDataset(dataset, processor, args.phase)
    
    checkpoint = torch.load(args.state_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
        
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=test_collator)
    inference(args, model, dataset, test_dataloader, processor, device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--img_path', type=str, default='./data/test/png')
    parser.add_argument('--table_path', type=str, default='./data/test/tables')
    parser.add_argument('--row_path', type=str, default='./data/test/gpt_indexes')
    parser.add_argument('--col_path', type=str, default='./data/test/gpt_columns')
    parser.add_argument('--json_path', type=str, default='./data/test/annotations')    
    
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--phase', type=int, default=3)
    
    parser.add_argument('--state_path', type=str, default='./state/phase_2_best_model.pth')
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--theta', type=float, default=0.5)
    
    args = parser.parse_args()
    
    main(args)
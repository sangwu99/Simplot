import argparse
import torch 
import gc 
import os 

import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup
from transformers import Pix2StructProcessor

from model import Simplot
from dataset import prepare_dataset, SimplotDataset, phase_1_collator, phase_2_collator
from trainer import train

def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    processor = Pix2StructProcessor.from_pretrained('google/deplot')
    processor.image_processor.is_vqa = True
    
    np.random.seed(args.seed)
    model_save_path = args.model_save_path
    os.makedirs(model_save_path, exist_ok = True)
    
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr 
    margin = args.margin
    gpu = args.device 
    lambda_ = args.lambda_
    
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    model = Simplot(args, margin, lambda_)

    train_dataset, test_dataset = prepare_dataset(args)
    train_dataset, test_dataset = SimplotDataset(train_dataset, processor, args.phase), SimplotDataset(test_dataset, processor, 1)

    if args.phase == 1:
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = phase_1_collator, drop_last = True)
        test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, collate_fn = phase_1_collator, drop_last = False)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = phase_2_collator, drop_last = True)
        test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, collate_fn = phase_1_collator, drop_last = False)

        checkpoint = torch.load(args.state_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    warm_up_steps = len(train_dataloader)
    num_training_steps = len(train_dataloader) * epochs
    
    optimizer = Adafactor(model.parameters(), lr = learning_rate, weight_decay = 1e-05, relative_step = False, scale_parameter = False)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = warm_up_steps, num_training_steps = num_training_steps)
    
    model.to(device)
    model.train()
    
    train(args, model, train_dataloader, test_dataloader, optimizer, scheduler, device)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--img_path', type=str, default='./data/train/png')
    parser.add_argument('--table_path', type=str, default='./data/train/tables')
    parser.add_argument('--row_path', type=str, default='./data/train/indexes')
    parser.add_argument('--col_path', type=str, default='./data/train/columns')
    
    parser.add_argument('--test_img_path', type=str, default='./data/val/png')
    parser.add_argument('--test_table_path', type=str, default='./data/val/tables')
    parser.add_argument('--test_row_path', type=str, default='./data/val/indexes')
    parser.add_argument('--test_col_path', type=str, default='./data/val/columns')
    
    parser.add_argument('--pos_img_path', type=str, default='./data/train/positive_png')
    parser.add_argument('--neg_img_path', type=str, default='./data/train/negative_png')
    parser.add_argument('--pos_test_img_path', type=str, default='./data/val/positive_png')

    parser.add_argument('--state_path', type=str, default='')
    parser.add_argument('--model_save_path', type=str, default='./state/')

    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--phase', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lambda_', type=float, default=0.9)
    parser.add_argument('--margin', type=float, default=0)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    main(args)
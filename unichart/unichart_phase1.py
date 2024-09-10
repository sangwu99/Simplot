import argparse
import torch
import os 
import re
import gc

from typing import List

from PIL import Image
from tqdm import tqdm
import pandas as pd

from transformers import DonutProcessor, VisionEncoderDecoderModel
from torch.utils.data import DataLoader
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup

from data import Phase1Dataset
from utils import prepare_dataset

EPOCHS = 10

def main():
  parser = argparse.ArgumentParser(description='Train Chart Transformer')
  parser.add_argument('--phase', type=int, default=4, help='Phase 1 or 2')
  parser.add_argument('--img_path', type=str, default="../examples/train_chartqa/png/", help='Path to the images')
  parser.add_argument('--table_path', type=str, default="../examples/train_chartqa/tables/", help='Path to the tables')
  parser.add_argument('--row_path', type=str, default="../examples/train_chartqa/indexes/", help='Path to the row indexes')
  parser.add_argument('--col_path', type=str, default="../examples/train_chartqa/columns/", help='Path to the column indexes')
  parser.add_argument('--pos_img_path', type=str, default="../examples/train_chartqa/1209positive/", help='Path to the positive images')
  
  parser.add_argument('--test_img_path', type=str, default="../examples/test/png/", help='Path to the test images')
  parser.add_argument('--test_table_path', type=str, default="../examples/test/tables/", help='Path to the test tables')
  parser.add_argument('--test_row_path', type=str, default="../examples/test/indexes/", help='Path to the test row indexes')
  parser.add_argument('--test_col_path', type=str, default="../examples/test/columns/", help='Path to the test column indexes')
  parser.add_argument('--pos_test_img_path', type=str, default="../examples/test/1222positive/", help='Path to the positive test images')
  parser.add_argument('--model_save_path', type=str, default="../state/unichart/phase1", help='Path to the model save directory')

  parser.add_argument('--batch-size', type=int, default=6, help='Batch Size for the model')
  parser.add_argument('--valid-batch-size', type=int, default=4, help='Valid Batch Size for the model')
  parser.add_argument('--max-length', type=int, default=1024, help='Max length for decoder generation')
  parser.add_argument('--num-workers', type=int, default=2, help='Number of workers')
  parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
  parser.add_argument('--gpus-num', type=int, default=7, help='gpus num')

  parser.add_argument('--checkpoint-path', type=str, default = "ahmed-masry/unichart-base-960", help='Path to the checkpoint')

  args = parser.parse_args()

  model_save_path = args.model_save_path
  os.makedirs(model_save_path, exist_ok=True)
  
  processor = DonutProcessor.from_pretrained(args.checkpoint_path)
  model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint_path)
  
  processor.tokenizer.add_tokens('<columns>')
  processor.tokenizer.add_tokens('<rows>')
  
  model.decoder.resize_token_embeddings(len(processor.tokenizer))
  dataset, test_dataset = prepare_dataset(args)

  train_dataset = Phase1Dataset(dataset = dataset, 
                                max_length=args.max_length, 
                                processor = processor)

  val_dataset = Phase1Dataset(dataset = test_dataset,
                              max_length=args.max_length, 
                              processor = processor)
  
  optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, lr=0.01, weight_decay=1e-05)
  scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=40000)
  
  device = f"cuda:{args.gpus_num}" 
  model.to(device)
  model.train() 
  
  losses = []
  
  train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
  val_dataloader = DataLoader(val_dataset, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.num_workers)
  
  best_test_loss = float('inf')
  best_epoch = 0
  
  for epoch in tqdm(range(EPOCHS)):
    torch.cuda.empty_cache()
    gc.collect()
        
    model.train()
        
    train_loss_list = []
    test_loss_list = []
    print("Epoch:", epoch)
    for idx, batch in enumerate(train_dataloader):
      gc.collect()
      pixel_values, decoder_input_ids, labels = batch
      pixel_values = pixel_values.to(device)
      decoder_input_ids = decoder_input_ids.to(device)
      labels = labels.to(device)
            
      outputs = model(pixel_values,
                      decoder_input_ids=decoder_input_ids[:, :-1],
                      labels=labels[:, 1:])
            
      print("Loss:", outputs.loss.item())
      train_loss_list.append(outputs.loss.item())

      outputs.loss.backward()

      optimizer.step()
      optimizer.zero_grad()
            
    scheduler.step()
    train_loss = sum(train_loss_list) / len(train_dataloader)
        
    del outputs  
    del pixel_values
    del decoder_input_ids
    del labels
            
    print(f'{epoch} epoch train finished')
    
    gc.collect()
    torch.cuda.empty_cache()
              
    model.eval() 
    for idx, batch in enumerate(val_dataloader):
        pixel_values, decoder_input_ids, labels = batch
        pixel_values = pixel_values.to(device)
        decoder_input_ids = decoder_input_ids.to(device)
        labels = labels.to(device)
        
        outputs = model(pixel_values,
                        decoder_input_ids=decoder_input_ids[:, :-1],
                        labels=labels[:, 1:])
        
        test_loss_list.append(outputs.loss.item())
        
    del outputs  
    del pixel_values
    del decoder_input_ids
    del labels
    
    print(f'{epoch} epoch test finished')
    
    test_loss = sum(test_loss_list) / len(val_dataloader)
    losses.append({'epoch' : epoch, 'train_loss': train_loss ,'test_loss' : test_loss})
    
    summary = pd.DataFrame(losses)
    summary.to_csv(f'{model_save_path}/summary.csv')
    
    torch.save(model.state_dict(), f'{model_save_path}/{epoch}_model_state_dict.pt')

    if test_loss < best_test_loss:
      best_test_loss = test_loss
      best_epoch = epoch
      torch.save({'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict()},
                  f'{model_save_path}/phase_{args.phase}_best_model.pth')

    
  torch.save(model.state_dict(), f'{model_save_path}/model_state_dict.pt')
  print(f"Best model saved with test loss {best_test_loss} at epoch {best_epoch}")


if __name__ == '__main__':
  main()
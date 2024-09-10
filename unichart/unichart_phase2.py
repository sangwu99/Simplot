import argparse
import torch
import os
import re
import gc 
import copy

from PIL import Image
from tqdm import tqdm
import pandas as pd

from transformers import DonutProcessor, VisionEncoderDecoderModel
from torch.utils.data import DataLoader
from typing import List

from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup

from data import Phase1Dataset, Phase2Dataset
from utils import prepare_dataset


EPOCHS = 10

def main():
  parser = argparse.ArgumentParser(description='Train Chart Transformer')
  parser.add_argument('--phase', type=int, default=5, help='Phase 1 or 2')
  parser.add_argument('--img_path', type=str, default="../data/train/png/", help='Path to the images')
  parser.add_argument('--table_path', type=str, default="../data/train/tables/", help='Path to the tables')
  parser.add_argument('--row_path', type=str, default="../data/train/indexes/", help='Path to the row indexes')
  parser.add_argument('--col_path', type=str, default="../data/train/columns/", help='Path to the column indexes')
  parser.add_argument('--pos_img_path', type=str, default="../data/train/positive_png/", help='Path to the positive images')
  parser.add_argument('--neg_img_path', type=str, default="../data/train/negative_png/", help='Path to the negative images')
  
  parser.add_argument('--test_img_path', type=str, default="../data/val/png/", help='Path to the test images')
  parser.add_argument('--test_table_path', type=str, default="../data/val/tables/", help='Path to the test tables')
  parser.add_argument('--test_row_path', type=str, default="../data/val/indexes/", help='Path to the test row indexes')
  parser.add_argument('--test_col_path', type=str, default="../data/val/columns/", help='Path to the test column indexes')
  parser.add_argument('--pos_test_img_path', type=str, default="../data/val/positive_png/", help='Path to the positive test images')
  parser.add_argument('--model_save_path', type=str, default="../state/unichart/phase2", help='Path to the model save directory')
  parser.add_argument('--margin', type=float, default=0, help='Margin for triplet loss')
  parser.add_argument('--lambda_', type=float, default=0.9, help='Lambda for loss')
  parser.add_argument('--model_state_dict', type=str, default='../state/unichart/phase1/phase_1_best_model.pth.pt', help='Path to the model state dict')

  parser.add_argument('--batch-size', type=int, default=3, help='Batch Size for the model')
  parser.add_argument('--valid-batch-size', type=int, default=4, help='Valid Batch Size for the model')
  parser.add_argument('--max-length', type=int, default=1024, help='Max length for decoder generation')
  parser.add_argument('--num-workers', type=int, default=2, help='Number of workers')
  parser.add_argument('--lr', type=float, default=1e-04, help='learning rate')
  parser.add_argument('--gpus-num', type=int, default=5, help='gpus num')

  parser.add_argument('--checkpoint-path', type=str, default = "ahmed-masry/unichart-base-960", help='Path to the checkpoint')

  args = parser.parse_args()

  model_save_path = args.model_save_path
  os.makedirs(model_save_path, exist_ok=True)
  
  processor = DonutProcessor.from_pretrained(args.checkpoint_path)
  model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint_path)
  
  processor.tokenizer.add_tokens('<columns>')
  processor.tokenizer.add_tokens('<rows>')
  
  model.decoder.resize_token_embeddings(len(processor.tokenizer))
  
  model.load_state_dict(torch.load(args.model_state_dict))
  model.teacher_encoder = copy.deepcopy(model.encoder)
  model.teacher_encoder.requires_grad_(False)
  model.decoder.requires_grad_(False)
  
  dataset, test_dataset = prepare_dataset(args)

  train_dataset = Phase2Dataset(dataset = dataset, 
                                max_length=args.max_length, 
                                processor = processor)

  val_dataset = Phase1Dataset(dataset = test_dataset,
                              max_length=args.max_length, 
                              processor = processor)
  print(dataset[0]['text'])
  print(dataset[0]['col'])
  print(dataset[0]['row'])
  
  device = f"cuda:{args.gpus_num}" 
  model.to(device)
  model.train() 
  
  losses = []
  batch_size = args.batch_size
  
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
  val_dataloader = DataLoader(val_dataset, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.num_workers)
  
  learning_rate = args.lr
  warm_up_steps = len(train_dataloader)
  num_training_steps = len(train_dataloader) * EPOCHS
  
  optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, lr=learning_rate, weight_decay=1e-05)
  scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps, num_training_steps=num_training_steps)
  
  CE_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
  TripletLoss = torch.nn.TripletMarginLoss(margin= args.margin)
  
  lambda_ = args.lambda_
  best_test_loss = float('inf')
  best_epoch = 0
  
  for epoch in tqdm(range(EPOCHS)):
    
    torch.cuda.empty_cache()
    gc.collect()
        
    model.train()
        
    train_loss_list = []
    test_loss_list = []
    hidden_state_loss_list = []
    decoder_loss_list = []
    print("Epoch:", epoch)
    
    for idx, batch in enumerate(train_dataloader):
      gc.collect()
      
      anchor_pixel_values, positive_pixel_values, negative_pixel_values, decoder_input_ids, labels = batch
      
      anchor_pixel_values = anchor_pixel_values.to(device)
      positive_pixel_values = positive_pixel_values.to(device)
      negative_pixel_values = negative_pixel_values.to(device)
      decoder_input_ids = decoder_input_ids.to(device)
      labels = labels.to(device)
      
      anchor = model.encoder(anchor_pixel_values)[0]
      hidden_states = model.teacher_encoder(torch.concat([positive_pixel_values, negative_pixel_values], dim=0))[0]
      
      decoder_outputs = model.decoder(decoder_input_ids[:, :-1].repeat(3, 1), 
                                      encoder_hidden_states=torch.concat([anchor, hidden_states], dim=0),
                                      encoder_attention_mask = None,
                                      output_hidden_states = True)
      
      logits = decoder_outputs.logits[:batch_size]
      decoder_loss = CE_loss(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
      hidden_state_loss = TripletLoss(decoder_outputs.hidden_states[0][:batch_size],
                                      decoder_outputs.hidden_states[0][batch_size:batch_size*2].detach(),
                                      decoder_outputs.hidden_states[0][batch_size*2:].detach())
      
      loss = (lambda_ * decoder_loss) + ((1 - lambda_) * hidden_state_loss)
      
      print("Hidden state loss:", hidden_state_loss.item())
      print("Decoder loss:", decoder_loss.item())
      print("Loss:", loss.item())
      train_loss_list.append(loss.item())
      hidden_state_loss_list.append(hidden_state_loss.item())
      decoder_loss_list.append(decoder_loss.item())
      
      loss.backward()

      optimizer.step()
      optimizer.zero_grad()
            
      scheduler.step()
    
    hidden_state_loss = sum(hidden_state_loss_list) / len(train_dataloader)
    decoder_loss = sum(decoder_loss_list) / len(train_dataloader)
    train_loss = sum(train_loss_list) / len(train_dataloader)
    
        
    del anchor_pixel_values
    del positive_pixel_values
    del negative_pixel_values
    del decoder_input_ids
    del labels
    del anchor
    del hidden_states
    del decoder_outputs
    del logits
    del loss
            
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
    losses.append({'epoch' : epoch, 'decoder_loss': decoder_loss, 'hidden_state_loss': hidden_state_loss , 'train_loss': train_loss ,'test_loss' : test_loss})
    
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
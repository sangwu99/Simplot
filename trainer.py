import torch 
import gc 
import os

import pandas as pd
from tqdm import tqdm

from utils.metric import get_rd


def train(args, model, train_dataloader, test_dataloader, optimizer, scheduler, device):
    losses = []
    epohcs = args.epochs
    best_test_loss = float('inf')
    best_epoch = 0
    
    for epoch in tqdm(range(epohcs)):
        torch.cuda.empty_cache()
        gc.collect()
        
        model.train()
                
        train_loss_list = []
        test_loss_list = []
        print("Epoch:", epoch)

        for idx, batch in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            gc.collect()
            labels = batch.pop("labels").to(device)
            flattened_patches = batch.pop("flattened_patches").to(device)
            attention_mask = batch.pop("attention_mask").to(device)
            
            if args.phase == 1:
                loss = model(flattened_patches = flattened_patches, 
                             attention_mask = attention_mask,
                             labels=labels)
            else:
                loss = model.forward_phase_2(flattened_patches = flattened_patches, 
                                             attention_mask = attention_mask,
                                             labels=labels,
                                             batch_size = args.batch_size)

            print("Loss:", loss.item())
            train_loss_list.append(loss.item())

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
        train_loss = sum(train_loss_list) / len(train_dataloader)
        
        del loss  
        del flattened_patches
        del attention_mask
        del labels
                    
        torch.cuda.empty_cache()
        gc.collect()
                  
        model.eval() 
        
        for idx, batch in enumerate(test_dataloader):
            labels = batch.pop("labels").to(device)
            flattened_patches = batch.pop("flattened_patches").to(device)
            attention_mask = batch.pop("attention_mask").to(device)
            
            loss = model(flattened_patches = flattened_patches, 
                            attention_mask = attention_mask,
                            labels=labels)
            test_loss_list.append(loss.item())
            
        del loss
        del flattened_patches
        del attention_mask
        del labels
                
        test_loss = sum(test_loss_list) / len(test_dataloader)
        losses.append({'epoch' : epoch, 'train_loss': train_loss ,'test_loss' : test_loss})
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()},
                       f'{args.model_save_path}/phase_{args.phase}_best_model.pth')
        
        summary = pd.DataFrame(losses)
        summary.to_csv(f'{args.model_save_path}/summary.csv')
        
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()},
                   f'{args.model_save_path}/phase_{args.phase}_{epoch}_model_state_dict.pth')

    print(f"Best model saved with test loss {best_test_loss} at epoch {best_epoch}")

def inference(args, model, dataset, dataloader, processor, device):
    
    if args.inference_type == 'QA':
        accuracy_list = {'img':[], 'type':[], 'pred':[], 'label':[]}

        for idx, batch in enumerate(dataloader):
            labels = batch.pop("labels").to(device)
            flattened_patches = batch.pop("flattened_patches").to(device)
            attention_mask = batch.pop("attention_mask").to(device)
            chart_type = batch.pop("type")
            
            predictions = model.generate(flattened_patches= flattened_patches,
                                        attention_mask=attention_mask,)    
            pred = processor.batch_decode(predictions, skip_special_tokens=True)
            label = processor.batch_decode(labels, skip_special_tokens=True)
            
            accuracy_list['img'].append(dataset[idx]['img_name'])
            accuracy_list['type'].append(chart_type[0])
            accuracy_list['pred'].append(pred[0])
            accuracy_list['label'].append(label[0])

        result_df = pd.DataFrame(accuracy_list)
        
        rd_df, failed = get_rd(result_df)
        rd_df.to_csv(os.path.join(args.result_path, 'prediction.csv'))
    
    else:
        accuracy_list = {'img':[], 'pred':[]}
        for idx, batch in enumerate(sample_dataloader):
            flattened_patches = batch.pop("flattened_patches").to(device)
            attention_mask = batch.pop("attention_mask").to(device)
            
            predictions = model.generate(flattened_patches= flattened_patches,
                                        attention_mask=attention_mask)    
            pred = processor.batch_decode(predictions, skip_special_tokens=True)
            
            accuracy_list['img'].append(dataset[idx]['img_name'])
            accuracy_list['pred'].append(pred[0])
            
        result_df = pd.DataFrame(accuracy_list)
        result_df.to_csv(os.path.join(args.result_path, 'opencqa_prediction.csv'))
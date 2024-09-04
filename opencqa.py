import argparse 
import json 

import pandas as pd 

from utils.llm_utils import pred_to_md, calculate_BLEU
from utils.llm_prompts import opencqa_prompt

def main(args):
    
    pred = pd.read_csv(args.prediction)
    pred['md'] = pred['pred'].apply(pred_to_md)
    args.pred = pred
    
    with open(args.metadata_path) as f:
        metadata = json.load(f)
        
    qa = pd.DataFrame.from_dict(metadata, orient='index', columns=['img','title','article','summary','question','abstractive_answer','extractive_answer'])
    args.qa = qa.reset_index().drop(columns=['index'])
    
    qa_list = []
    
    for index in range(len(args.qa)):
        try:
            qa_list.append(opencqa_prompt(args, index))
        except:
            print(f"Error in {index}")
            continue

    qa_results = pd.DataFrame(qa_list)
    qa_results = calculate_BLEU(qa_results)
    qa_results.to_csv('./results/opencqa_results.csv', index=False)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='opencqa')
        
    parser.add_argument('--prediction', type=str, default='./results/opencqa_prediction.csv', help='Path to prediction.csv')
    parser.add_argument('--img_path', type=str, default='./data/opencqa/test/img/', help='Path to chart images')
    parser.add_argument('--metadata_path', type=str, default='./data/opencqa/etc/data(full_summary_article)/test_extended.json', help='Path to json files')
    
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature')
    parser.add_argument('--api_key', type=str, default='', help='Your OpenAI API key')
    
    args = parser.parse_args()
    
    main(args)
    
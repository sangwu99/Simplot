import argparse 
import json 

import pandas as pd 

from utils.llm_utils import pred_to_md, calculate_QA
from utils.llm_prompts import prompt

def main(args):
    
    if args.qa_type == 'human':
        qa_path = f"./data/test/test_human.json"
    elif args.qa_type == 'augmented':
        qa_path = f"./data/test/test_augmented.json"
    else:
        raise ValueError('QA type not recognized')
    
    args.qa = json.load(open(qa_path))
    
    pred = pd.read_csv(args.prediction)
    pred['md'] = pred['pred'].apply(pred_to_md)
    args.pred = pred
    
    qa_list = []
    
    for index in range(len(args.qa)):
        try:
            qa_list.append(prompt(args, index))
        except:
            print(f"Error in {index}")
            continue

    qa_results = pd.DataFrame(qa_list)
    qa_results, errors = calculate_QA(qa_results)
    qa_results.to_csv('./results/qa_results.csv', index=False)
    
    print(f"Total Errors: {len(errors)}")
    print(f"Total Questions: {len(qa_results)}")
    print(f"Accuracy: {qa_results['correct'].mean()}")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='QA')
    
    parser.add_argument('--qa_type', type=str, default='human', help='Type of QA')
    
    parser.add_argument('--prediction', type=str, default='./results/prediction.csv', help='Path to prediction.csv')
    parser.add_argument('--img_path', type=str, default='./data/test/png/', help='Path to chart images')
    parser.add_argument('--json_path', type=str, default='./data/test/annotations/', help='Path to json files')
    
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature')
    parser.add_argument('--api_key', type=str, default='', help='Your OpenAI API key')
    
    args = parser.parse_args()
    
    main(args)
    
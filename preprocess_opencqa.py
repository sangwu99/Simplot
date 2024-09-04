import os
import ast
import json
import shutil

from tqdm import tqdm


def main(args):
    
    opencqa_path = './data/opencqa/'
    chart_base_path = os.path.join(opencqa_path,'chart_images')
    
    with open(os.path.join(opencqa_path, 'etc/data(full_summary_article)/test_extended.json')) as f:
        metadata = json.load(f)    
    
    test_idx = list(metadata.keys())
    
    for item in tqdm(test_idx):
        img_path = os.path.join(chart_base_path,f"{item}.png")
        img_arrive_path = f'./data/opencqa/test/img/{item}.png'
        
        shutil.copy(img_path, img_arrive_path)
    
if __name__ == '__main__':
    main()
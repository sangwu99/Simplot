
import os 
import base64
import json
import openai

def prompt(args, index):
    openai.api_key = args.api_key
    
    img = args.qa[index]['imgname']
    image_path = os.path.join(args.img_path, img)
    
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    with open(os.path.join(args.json_path, img[:-3]+'json'),'r') as f:
        data_json = json.load(f)
    
        
    prompt = f'''
    The first image is one of a bar chart, line chart, or pie chart.
    It comes with instructions, a predicted table, a question, and an answer. 
    Based on the image and its predicted table, the question must be solved.
    You must give your response in a structured JSON format that can be directly parsed with json.loads.
    
    Your response should contain four fields and four fields only: 
    "instruction explanation": an explanation the process of following instructions according to the chart type
    "explanation": an explanation of how you arrived at the answer
    "answer": the answer based on the chart and question

    Instruction for whole chart: 
    1. If the question is about values corresponding to specific positions (ex: lower, last, top), then you must match the table with the chart image's positions for reasoning. 
    2. If the question demands interpretation based on the magnitude of values, reasoning should be based on the table's values. 
    3. Originally, this task requires solving based solely on the image, meaning all positions should be interpreted based on the image itself.
    4. In most cases, the presence of x-axis values or y-axis values enables the determination of the chart's values.
    5. Note that, you can utilize the information from the predicted table. The predicted columns and rows are very likely to correspond to the actual columns and rows of the chart, and this can help you determine where the rows and columns exist in the chart image.
    
    Instruction for bar chart:
    1. Firstly, bars of the same color represent the same column. Therefore, distinguishing colors and identifying corresponding columns is crucial (usually displayed around the main chart in the form of a legend).
    2. Next, determine the location of rows. For vertical bar charts, rows are typically annotated at the bottom of the main chart, while for horizontal bar charts, they are annotated on the left or right side of the main chart.
    3. Then, combine the colors of the nearest bars with annotated rows to determine which row and column the bars correspond to in the table.
    4. Afterwards, locate the values corresponding to each row and column. If values are annotated on the bars, refer to them. Otherwise, compare the sizes of the bars to find the values.
    5. For vertical bar charts, the y-axis value at the end of the bar corresponds to its value. Similarly, for horizontal bar charts, the x-axis value at the end of the bar corresponds to its value.
    
    Instruction for line chart: 
    1. In the case of a line chart, the bottom x-axis will primarily represent the rows, and each colored line will represent a column.
    2. The legend, which indicates which column corresponds to the color of the line, is usually located within the main chart. If the legend is absent or located separately, the text annotated with the color corresponding to the line will likely indicate the column (if colors are not present, the text annotated at the left or right end of the line is likely to correspond to the column).
    3. The point of the line passing through the same x coordinates as each x-axis is the value itself (meaning the x-axis corresponds to the row, the color of the line corresponds to the column, and that point is the value).
    4. If there is an annotation near a line point, it is highly likely that this value represents the value of the point.
    5. If there is no annotation near a line point, you can determine the value of the point by referring to the y-axis value corresponding to the y coordinate of the point.
    6. In a line chart, it is crucial to understand the flow of the line. Lines can show trends of decreasing, increasing, or remaining constant, and when multiple lines intersect, it is important to identify which line corresponds to which column based on their colors.
    
    Instruction for pie chart:
    1. In a pie chart, it is very important to determine which color corresponds to which row.
    2. Each section has one color, and the row it corresponds to is likely indicated by text either inside the section or close to it (if not nearby, it can be identified through the legend or connected to the corresponding text by lines or markers).
    3. In the case of a pie chart, the values are usually annotated on each section of the pie chart.
    
    Now you have to answer the question based on the first image, predicted table and question.
    
    Please follow the below rules:
    1. Ensure your response excludes "```json" to allow for immediate parsing via json.loads
    2. If question ask a yes/no, then answer should be "Yes" or "No". You must not answer with "true" or "false".
    3. At 'answer' field, you must only answer the question. You must not include any other information.

    Chart image is first image.
    
    Here is a table:
    {args.pred[args.pred['img'] == args.qa[index]['imgname']]['md'].values[0]}

    Here is a question:
    {args.qa[index]['query']}
    
    A:
    '''
    
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                },
            ]
        }
    ]
    params = {
    "model": "gpt-4-vision-preview",
    "messages": prompt_messages,
    "api_key": openai.api_key,
    "max_tokens": 4096,
    'temperature': args.temperature,
    }
    try:
        result = openai.ChatCompletion.create(**params)
        return {'index':index,
                'img': img,
                'type': data_json['type'],
                'query': args.qa[index]['query'],
                'answer': result.choices[0].message.content, 
                'label': args.qa[index]['label']}
    except: 
        raise ValueError('GPT-4 failed to generate response. Please try again.')
    

def opencqa_prompt(args, index):
    openai.api_key = args.api_key
    
    img = args.qa['imgname'][index]
    image_path = os.path.join(args.img_path, img)
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    table = args.pred.loc[args.pred['img']==img,'md'].values[0]
    question = args.qa['question'][index]
    ground_truth = args.qa['abstractive_answer'][index]
        
    prompt = f'''
    The first image is one of a bar chart, line chart, or pie chart.
    It comes with instructions, a predicted table, a question, and an answer. 
    Based on the image and its predicted table, provide an open-ended answer to the following question.

    Instruction for whole chart: 
    1. If the question is about values corresponding to specific positions (ex: lower, last, top), then you must match the table with the chart image's positions for reasoning. 
    2. If the question demands interpretation based on the magnitude of values, reasoning should be based on the table's values. 
    3. Originally, this task requires solving based solely on the image, meaning all positions should be interpreted based on the image itself.
    4. In most cases, the presence of x-axis values or y-axis values enables the determination of the chart's values.
    5. Note that, you can utilize the information from the predicted table. The predicted columns and rows are very likely to correspond to the actual columns and rows of the chart, and this can help you determine where the rows and columns exist in the chart image.

    Instruction for bar chart:
    1. Firstly, bars of the same color represent the same column. Therefore, distinguishing colors and identifying corresponding columns is crucial (usually displayed around the main chart in the form of a legend).
    2. Next, determine the location of rows. For vertical bar charts, rows are typically annotated at the bottom of the main chart, while for horizontal bar charts, they are annotated on the left or right side of the main chart.
    3. Then, combine the colors of the nearest bars with annotated rows to determine which row and column the bars correspond to in the table.
    4. Afterwards, locate the values corresponding to each row and column. If values are annotated on the bars, refer to them. Otherwise, compare the sizes of the bars to find the values.
    5. For vertical bar charts, the y-axis value at the end of the bar corresponds to its value. Similarly, for horizontal bar charts, the x-axis value at the end of the bar corresponds to its value.

    Instruction for line chart: 
    1. In the case of a line chart, the bottom x-axis will primarily represent the rows, and each colored line will represent a column.
    2. The legend, which indicates which column corresponds to the color of the line, is usually located within the main chart. If the legend is absent or located separately, the text annotated with the color corresponding to the line will likely indicate the column (if colors are not present, the text annotated at the left or right end of the line is likely to correspond to the column).
    3. The point of the line passing through the same x coordinates as each x-axis is the value itself (meaning the x-axis corresponds to the row, the color of the line corresponds to the column, and that point is the value).
    4. If there is an annotation near a line point, it is highly likely that this value represents the value of the point.
    5. If there is no annotation near a line point, you can determine the value of the point by referring to the y-axis value corresponding to the y coordinate of the point.
    6. In a line chart, it is crucial to understand the flow of the line. Lines can show trends of decreasing, increasing, or remaining constant, and when multiple lines intersect, it is important to identify which line corresponds to which column based on their colors.

    Instruction for pie chart:
    1. In a pie chart, it is very important to determine which color corresponds to which row.
    2. Each section has one color, and the row it corresponds to is likely indicated by text either inside the section or close to it (if not nearby, it can be identified through the legend or connected to the corresponding text by lines or markers).
    3. In the case of a pie chart, the values are usually annotated on each section of the pie chart.

    Now you have to answer the question based on the first image, predicted table and question.

    Please follow the below rules:
    1. You should only provide open-ended answers. 
    2. Open-ended answers should preferably be within 55 words. 

    Here is a table:
    {table}

    Here is a question:
    {question}

    A:
    '''
 
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                },
            ]
        }
    ]
    params = {
    "model": "gpt-4-turbo",
    "messages": prompt_messages,
    "api_key": openai.api_key,
    "max_tokens": 4096,
    'temperature': args.temperature,
    }
    try:
        result = openai.ChatCompletion.create(**params)
        return {'index':index,
                'img': img,
                'question': question, 
                'ground_truth': ground_truth,
                'answer': result.choices[0].message.content}
    except: 
        raise ValueError('GPT-4 failed to generate response. Please try again.')
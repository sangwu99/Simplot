
import re
import json
import sacrebleu

def pred_to_md(pred):
    pred = pred.split('<0x0A>')
    new_pred = []
    for i in range(len(pred)):
        if i == 0:
            new_pred.append('Header: ' + pred[i])
        else:
            new_pred.append(f'Row {i}: '+ pred[i])

    new_pred = '\n'.join(new_pred)
    return new_pred

def extract_numbers(s):
    return ''.join(re.findall(r'\d', s))

def extract_numbers_and_dots(input_string):
    numbers_and_dots = re.findall(r'[\d.]+', input_string)
    return ''.join(numbers_and_dots)

def calculate_ratio(ratio_string):
    numerator, denominator = ratio_string.split(':')
    numerator = float(numerator)
    denominator = float(denominator)
    result = numerator / denominator
    return result

def compare_pred_label(pred, label):
    label_list = re.sub(r'[\[\]]', '', label).split(', ')
    pred_list = [item.strip() for item in re.split(r' and |, ', pred)]
    
    label_list.sort()
    pred_list.sort()
    
    return label_list == pred_list
        
def get_lrstrip(text):
    text = str(text)
    text = text.rstrip()
    text = text.lstrip()

    return text.lower()

def _to_float(text):
    if text.endswith("%"):
        return float(text.rstrip('%'))
    else:
        return float(text)
    
def get_prediction(pred, label):
    pred = get_lrstrip(pred)
    label = get_lrstrip(label)
    if label.endswith("]"):
        return compare_pred_label(pred, label)
    try:
        label_float = _to_float(label)
        try:
            pred_float = _to_float(pred)

            relative_change = abs(pred_float - label_float) / abs(label_float)
                
            return relative_change <= 0.05
        except:
            try:
                pred_float = calculate_ratio(pred)
                relative_change = abs(pred_float - label_float) / abs(label_float)
                
                return relative_change <= 0.05
                
            except:
                pred_float = extract_numbers_and_dots(pred)
                pred_float = _to_float(pred_float)
                
                relative_change = abs(pred_float - label_float) / abs(label_float)
                return relative_change <= 0.05
    except:
        if pred == label:
            return True

        else:
            return False
        
def calculate_QA(df):
    error_list = []
    for i in range(len(df)):
        try:
            df.loc[i,'pred'] = json.loads(df['answer'][i])['answer']
        except:
            error_list.append(i)
    for i in range(len(df)):
        try:
            df.loc[i,'correct'] = get_prediction(df['pred'][i], df['label'][i])
        except:
            error_list.append(i)
    return df, error_list


def get_bleu(row):
    return sacrebleu.sentence_bleu(row['answer'],[row['ground_truth']], lowercase=True).score

def calculate_BLEU(df):
    df['bleu'] = df.apply(get_bleu, axis=1)
    print(f'\nBLEU: {result_df["bleu"].mean()}')

    return df
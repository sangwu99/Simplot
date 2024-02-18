import numpy as np

def get_lrstrip(text):
    text = str(text)
    text = text.rstrip()
    text = text.lstrip()
    text = text.rstrip('%')
    if text.endswith(".0"):
        text = text[:-2]
    
    return text

def get_table(csv):
    new_label = ''
    for idx in range(len(csv)):
        values = csv.values[idx]
        if idx==0:
            for i in range(len(values)):
                if i==0:
                    new_label += values[i]
                else:
                    new_label += f' | {values[i]}'
        else:
            for i in range(len(values)):
                if i==0:
                    new_label += f' <0x0A> {values[i]}'
                else:
                    new_label += f' | {values[i]}'
    return new_label

def rlstrip_4_column(csv):
    if len(csv.columns) > 1:
        text_list = []
        for idx in range(len(csv.columns)):
            text = str(csv[idx][0])
            text = text.rstrip()
            text = text.lstrip()
            text_list.append(text)
        return " | ".join(text_list)
    else:
        text = str(csv[0][0])
        text = text.rstrip()
        text = text.lstrip()
        return text
    
def rlstrip_4_index(csv):
    text_list = []
    for idx in range(len(csv.index)):
        text = str(csv[0][idx])
        text = text.rstrip()
        text = text.lstrip()
        text_list.append(text)
    return " | ".join(text_list)

def get_is_nan(text):
    try:
        return float(text)
    except:
        return np.nan
    
    

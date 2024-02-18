import numpy as np
import pandas as pd 

from scipy.optimize import linear_sum_assignment
from Levenshtein import distance

from utils.utils import get_lrstrip, get_is_nan

def normalized_levenshtein_distance(str1, str2, tau):
    distance_value = distance(str1, str2) / max(len(str1), len(str2))
    if distance_value > tau:
        return 1.0
    else:
        return distance_value

def relative_distance(p, t, theta):
    if t!=0:
        distance_value = min(1, abs((p - t) / t))
    else:
        distance_value = min(1, abs(p - t))
    if distance_value > theta:
        return 1.0
    else:
        return distance_value

def compute_rms_entry(p, t, tau, theta):
    similarity = (1 - normalized_levenshtein_distance(p[0] + p[1], t[0] + t[1], tau)) * (1 - relative_distance(p[2], t[2], theta))
    return similarity

def compute_relative_distance(p, t, theta):
    similarity = (1 - relative_distance(p[2], t[2], theta))
    return similarity

def custom_metric(pred, label, tau=1, theta=0.5):
    label_columns = []
    label_rows = [] 
    label_values = {}
    pred_columns = []
    pred_rows = [] 
    pred_values = {}
    pred_list = [] 
    label_list = []
    
    pred = pred.split('<0x0A>')
    label = label.split('<0x0A>')
    pred_col, pred_row = pred[0].split('|'), pred[1:]
    label_col, label_row = label[0].split('|'), label[1:]
    
    for i in label_col:
        label_columns.append(get_lrstrip(i))
    
    for i in pred_col:
        pred_columns.append(get_lrstrip(i))
        
    for i in label_row:
        i = i.split('|')
        tmp_dict = {}
        for j in range(len(i)):
            if j==0:
                tmp_row = get_lrstrip(i[j])
                label_rows.append(tmp_row)
            else:
                tmp_dict[label_columns[j]] = get_is_nan(get_lrstrip(i[j]))
                label_list.append((tmp_row, label_columns[j], get_is_nan(get_lrstrip(i[j]))))
        label_values[tmp_row] = tmp_dict 
        
    for i in pred_row:
        i = i.split('|')
        tmp_dict = {}
        for j in range(len(i)):
            if j==0:
                tmp_row = get_lrstrip(i[j])
                pred_rows.append(tmp_row)
            elif j >= len(pred_columns):
                continue
            else:
                tmp_dict[pred_columns[j]] = get_is_nan(get_lrstrip(i[j]))
                pred_list.append((tmp_row, pred_columns[j], get_is_nan(get_lrstrip(i[j]))))
        pred_values[tmp_row] = tmp_dict
    
    rd = compute_rd(pred_list, label_list, tau, theta)
    
    return rd

def compute_rd(p, t, tau, theta):

    N = len(p)
    M = len(t)
    
    if N == 0 or M == 0:
        return ZeroDivisionError

    similarity_matrix = np.zeros((N, M))
    relative_distance_matrix = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            similarity_matrix[i][j] = compute_rms_entry(p[i], t[j], tau, theta)
            relative_distance_matrix[i][j] = compute_relative_distance(p[i], t[j], theta)

    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

    rd_precision = np.sum(relative_distance_matrix[row_ind, col_ind]) / N
    rd_recall = np.sum(relative_distance_matrix[row_ind, col_ind]) / M

    if rd_precision + rd_recall == 0:
        rd_f1_score = 0
    else:
        rd_f1_score = (2 * (rd_precision * rd_recall)) / (rd_precision + rd_recall)

    return rd_f1_score

def get_rd(df):
    failed = []
    dataset = {'img':[], 'pred':[],'type':[], 'rd':[], 'label':[]}
    for i in range(len(df)):
        try:
            img = df['img'][i]
            rd = custom_metric(df['pred'][i], df['label'][i])
            if rd == ZeroDivisionError:
                rd = 0
            dataset['img'].append(img)
            dataset['pred'].append(df['pred'][i])
            dataset['type'].append(df['type'][i])
            dataset['label'].append(df['label'][i])
            dataset['rd'].append(rd)
        except:
            failed.append(i)
            
    rd = pd.DataFrame(dataset)
    rd.to_csv('rd.csv', index=False)
    print(rd.groupby('type')['rd'].mean())
    
    return rd, failed
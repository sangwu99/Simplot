import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import json

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
    

def extract_column_row(table, column_path, index_path, table_path):
    gt = pd.read_csv(os.path.join(table_path,table))
    columns = gt.columns[1:]
    indexes = gt.iloc[:,0]
    pd.DataFrame(columns.to_list()).T.to_csv(path_or_buf=os.path.join(column_path,table),header=False, index=False)
    indexes.to_csv(path_or_buf=os.path.join(index_path,table),header=False, index=False)
    


def modify_string(input_string, max_length=16):
    words = input_string.split()
    
    modified_lines = []
    current_line = words[0]
    
    for word in words[1:]:
        if len(current_line) + len(word) <= max_length:
            current_line += ' ' + word
        else:
            modified_lines.append(current_line.strip())
            current_line = word
    
    if current_line:
        modified_lines.append(current_line.strip())
    
    modified_result = '\n'.join(modified_lines)
    
    return modified_result


def get_strip(list, max_length=16):
    new_list = [] 
    for i in list:
        i = str(i)
        if len(i) > max_length:
            i = modify_string(i, max_length)
        new_list.append(i)
    return new_list

def replace_bad_word(x):
    try:
        return float(x.replace('%',''))
    except:
        return x

def custom_floor(number, decimal_places):
    factor = 10 ** decimal_places
    rounded_number = round(number * factor) / factor
    return rounded_number


def before_annotate(x):
    x = custom_floor(x, 2)
    x = str(x)
    if x.endswith('.0'):
        x = x[:-2]
    return x

def get_max_length(list):
    max_length = 0
    for i in list:
        i = str(i)
        if len(i) > max_length:
            max_length = len(i)
    return max_length


def preprocess_df_for_matplotlib(table, table_base_path, json_path):
    df = pd.read_csv(os.path.join(table_base_path, table))
    with open(os.path.join(json_path, table[:-3]+'json'),'r') as f:
        data_json = json.load(f)
    
    df.index = df[df.columns[0]].to_list()
    label_name = df.columns[0]
    df = df.drop(df.columns[0], axis=1)
    columns = df.columns.to_list()
    rows = df.index.to_list()
    for column in columns:
        df[column] = df[column].apply(replace_bad_word)
        
    shuffled_values = df.values.flatten()
    np.random.shuffle(shuffled_values)
    
    negative_df = pd.DataFrame(shuffled_values.reshape(df.shape), index=df.index, columns=df.columns)
        
    n_columns = len(columns)
    n_rows = len(rows)
    
    row_values = [df.loc[i].values for i in rows]
    col_values = [df[i].values for i in columns]
    
    negative_row_values = [negative_df.loc[i].values for i in rows]
    negative_col_values = [negative_df[i].values for i in columns]
    
    ind = np.arange(n_rows)
    
    return df, columns, rows, row_values, col_values, negative_row_values, negative_col_values, ind, n_columns, n_rows, label_name, data_json


def get_h_bar_matplotlib(columns, rows, row_values, ind, n_columns, n_rows, label_name, save_path=None,mode='save'):
    if n_columns * n_rows > 45:
        fig, ax = plt.subplots(figsize=(14, 12))
    elif n_columns * n_rows > 20:
        fig, ax = plt.subplots(figsize=(13, 11))
    elif n_columns * n_rows > 10:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    if n_columns * n_rows > 50:
        bar_height = 0.08
    elif n_columns * n_rows > 8:
        bar_height = 0.1
    else:
        bar_height = 0.15

    for i, entity in enumerate(columns):
        bars = []
        pos = ind - bar_height*(((n_columns-1) * 0.5)-i)
        values = [row[i] for row in row_values]
        bars += plt.barh(pos, values, bar_height, label=entity)
        for bar in bars:
            ax.annotate(before_annotate(bar.get_width()), 
                        xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2), 
                        xytext=(0, 3), 
                        textcoords="offset points", 
                        ha='center', 
                        va='bottom',
                        fontsize = 8)
        
    ax.set_yticks(ind)
    if get_max_length(rows) > 60:
        ax.set_yticklabels(get_strip(rows), fontsize = 7)
    else:
        ax.set_yticklabels(get_strip(rows))
    ax.invert_yaxis() 
    
    plt.ylabel(label_name, fontsize=10, labelpad=20, weight='bold')
    ax.legend(loc = 'upper left',bbox_to_anchor=(1.0, 1.0), fontsize=9)
    
    if mode == 'save':
        plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.clf()   # clear the current figure
        plt.close("all") # closes the current figure
    elif mode == 'show':
        plt.show()
    else:
        pass
    

    
def get_v_bar_matplotlib(columns, rows, row_values, ind, n_columns, n_rows, label_name,save_path=None,mode='save'):
    if n_columns * n_rows > 45:
        fig, ax = plt.subplots(figsize=(20, 8))
    elif n_columns * n_rows > 20:
        fig, ax = plt.subplots(figsize=(18, 8))
    elif n_columns * n_rows > 10:
        fig, ax = plt.subplots(figsize=(16, 8))
    else:
        fig, ax = plt.subplots(figsize=(14, 7))
    
    if n_columns * n_rows > 45:
        width = 0.08
    elif n_columns * n_rows > 8:
        width = 0.1
    else:
        width = 0.15
    
    for i, entity in enumerate(columns):
        bars = []
        pos = ind - width*(((n_columns-1)*0.5)-i)
        values = [row[i] for row in row_values]
        bars += ax.bar(pos, values, width, label=entity)
        
        for bar in bars:
            ax.annotate(before_annotate(bar.get_height()), 
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                        xytext=(0, 3), 
                        textcoords="offset points", 
                        ha='center', 
                        va='bottom',
                        fontsize = 8)


    # Set the title and labels
    ax.set_xticks(ind)
    if get_max_length(rows) > 60:
        ax.set_xticklabels(get_strip(rows, 8), fontsize = 7)
    elif n_rows > 7:
        ax.set_xticklabels(get_strip(rows, 8), fontsize = 7)
    else:
        ax.set_xticklabels(get_strip(rows, 8))
        
    plt.xlabel(label_name, fontsize = 10, rotation=0, labelpad=20, weight='bold')

    ax.legend(loc = 'upper left',bbox_to_anchor=(1.0, 1.0), fontsize=9)
    
    if mode=='save':
        plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.clf()   # clear the current figure
        plt.close("all") # closes the current figure
    elif mode=='show':
        plt.show()
    else:
        pass
    
def text_flag(text):
    try:
        text = int(text)
        return True 
    except:
        return False

def line_flag(table):
    flag = table.split('.')[0]
    
    if ('OECD' in flag) or ((len(flag) > 10) and text_flag(flag)):
        return True
    else:
        return False
    
def get_line_matplotlib(csv, positive_path, negative_path, table):
    flag = table.split('.')[0]
    
    if line_flag(flag):
        df = csv.copy()
        df = df.transpose()
        df.rename(columns = df.iloc[0], inplace=True)
        df = df.drop(df.index[0])
        df = df.replace('%','',regex=True)
        df = df.replace('\$','',regex=True)
        df = df.applymap(lambda x: pd.to_numeric(x, errors='coerce')) 
        df.index = df.index.astype(str) 
                
        plt.clf()
        plt.plot(df, marker='o')
        plt.grid()
        plt.legend(df.columns,loc ='upper left', bbox_to_anchor=(1.0, 1.0))
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                try:
                    plt.text(df.index[i], df.iloc[i,j], f'{round(df.iloc[i,j])}', ha='center', va='bottom')
                except:
                    pass
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(positive_path,table[:-3]+'png' ), format='png', bbox_inches='tight', pad_inches=0.1)
            
        # Permutation for negative samples
        for column in df.columns:
            df[column] = np.random.permutation(df[column].values)
        
        plt.clf()
        plt.plot(df, marker='o')
        plt.grid()
        plt.legend(df.columns,loc ='upper left', bbox_to_anchor=(1.0, 1.0))
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                try:
                    plt.text(df.index[i], df.iloc[i,j], f'{round(df.iloc[i,j])}', ha='center', va='bottom')
                except:
                    pass
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(negative_path,table[:-3]+'png' ), format='png', bbox_inches='tight', pad_inches=0.1)
        
    else:
        df = csv.copy()
        df.index = df.iloc[:,0].to_list()
        df = df.drop(df.columns[0],axis=1)
        df = df.replace('%','',regex=True) 
        df = df.replace('\$','',regex=True)
        df = df.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
        df.index = df.index.astype(str) 
            
        plt.clf()
        plt.plot(df, marker='o')
        plt.grid()
        plt.legend(df.columns,loc ='upper left', bbox_to_anchor=(1.0, 1.0))
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                try:
                    plt.text(df.index[i], df.iloc[i,j], f'{round(df.iloc[i,j])}', ha='center', va='bottom')
                except:
                    pass
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(positive_path,table[:-3]+'png' ), format='png', bbox_inches='tight', pad_inches=0.1)

        # Permutation for negative samples
        for column in df.columns:
            df[column] = np.random.permutation(df[column].values)
        
        plt.clf()
        plt.plot(df, marker='o')
        plt.grid()
        plt.legend(df.columns,loc ='upper left', bbox_to_anchor=(1.0, 1.0))
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                try:
                    plt.text(df.index[i], df.iloc[i,j], f'{round(df.iloc[i,j])}', ha='center', va='bottom')
                except:
                    pass
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(negative_path,table[:-3]+'png' ), format='png', bbox_inches='tight', pad_inches=0.1)

def get_pie_matplotlib(n_columns, n_rows, rows, col_values, save_path, table):
    if n_columns * n_rows > 40:
        fig, ax = plt.subplots(figsize=(12, 8))
    elif n_columns * n_rows > 20:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig, ax = plt.subplots()

    labels = get_strip(rows)
    sizes = col_values[0]
    def absolute_value(val):
        a  = np.round(val/100.*sizes.sum(), 0)
        return a
    wedges, _, autotexts = ax.pie(sizes, labels=labels, autopct=absolute_value, startangle=90)
    ax.axis('equal')  

    plt.legend(loc = 'upper left',bbox_to_anchor=(1.0, 1.0))
    plt.savefig(os.path.join(save_path,table[:-3]+'png'), format='png', bbox_inches='tight', pad_inches=0.1)

    plt.clf()   # clear the current figure
    plt.close("all") # closes the current figure 
    

def generate_simple_chart(table, table_base_path, json_path, positive_path, negative_path):
    df, columns, rows, row_values, col_values, negative_row_values, negative_col_values, ind, n_columns, n_rows, label_name, data_json = preprocess_df_for_matplotlib(table, table_base_path, json_path)
    if data_json['type'] == 'h_bar':
        get_h_bar_matplotlib(columns, rows, row_values, ind, n_columns, n_rows, 
                             label_name, save_path=os.path.join(positive_path, table[:-3]+'png'))
        get_h_bar_matplotlib(columns, rows, negative_row_values, ind, n_columns, n_rows, 
                             label_name, save_path=os.path.join(negative_path, table[:-3]+'png'))
    elif data_json['type'] == 'v_bar':
        get_v_bar_matplotlib(columns, rows, row_values, ind, n_columns, n_rows, 
                             label_name, save_path=os.path.join(positive_path, table[:-3]+'png'))
        get_v_bar_matplotlib(columns, rows, negative_row_values, ind, n_columns, n_rows, 
                             label_name, save_path=os.path.join(negative_path, table[:-3]+'png'))
    elif data_json['type'] == 'line':
        df = pd.read_csv(os.path.join(table_base_path, table))
        get_line_matplotlib(df, positive_path, negative_path ,table)

    elif data_json['type'] == 'pie':
        get_pie_matplotlib(n_columns, n_rows, rows, 
                           col_values, positive_path, table)
        get_pie_matplotlib(n_columns, n_rows, rows, 
                           negative_col_values, negative_path, table)
    else:
        raise ValueError('Invalid chart type')

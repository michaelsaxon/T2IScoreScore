import pandas as pd
import numpy as np

def robust_csv_line_split(line):
    line = line.strip()
    if '"' in line:
        # one or more cells contains commas, you have to split on the quotes first
        line = line.split('",')
        newline = []
        for elem in line:
            # if this is a quote line, we have already split it.
            # remove the other quote and put the whole thing on the list.
            # Else it's a block of comma separated values and should be split
            # this only works because none of my csvs lines end with strings
            if '"' in elem:
                assert elem.count('"') == 1
                if elem[0] == '"':
                    newline.append(elem[1:])
                else:
                    elem = elem.split(',"')
                    newline += elem
            else:
                newline += elem.split(",")
    else:
        newline = line.split(",")
    return newline

def slug_fname(fname):
    return ".".join(fname.split(".")[:-1])

def load_kvp_single_file(fname, convert_float = False, score_idx = 2, do_fname_slug = False):
    output_dict = {}
    with open(fname, "r") as f:
        lines = f.readlines()
    for line in lines[1:]:
        line = robust_csv_line_split(line)
        fname_slug = line[0]
        if do_fname_slug:
            fname_slug = slug_fname(fname_slug)
        score = line[score_idx]
        if convert_float:
            score = float(line[score_idx])
        output_dict[fname_slug] = score
    return output_dict

def load_kvp_multiple_files(fname, convert_float = False):
    output_dict = {}
    with open(fname, "r") as f:
        lines = list(map(robust_csv_line_split, f.readlines()))
    keys = lines[0]
    print(keys)
    for metric_name in keys[1:]:
        output_dict[metric_name] = {}
    for line in lines[1:]:
        # remove extension
        fname_slug = slug_fname(line[0])
        for i in range(1,len(line)):
            val = line[i]
            if val != '' and convert_float:
                val = float(val)
            if i > 6:
                print(line)
            output_dict[keys[i]][fname_slug] = val
    return output_dict

# 3 is index for ranks
# 0? is index for ids
def load_col_source_csv(fname, col_idx = 3):
    output_dict = {}
    with open(fname, "r") as f:
        lines = list(map(robust_csv_line_split, f.readlines()))
    for line in lines:
        fname_slug = slug_fname(line[2])
        rank = line[col_idx]
        output_dict[fname_slug] = rank
    return output_dict

def get_row(output_dict, fname_slug, tty = True):
    keys = output_dict.keys()
    out_string = []
    if tty:
        print(fname_slug)
    for key in keys:
        value = output_dict[key].get(fname_slug, '')
        if tty:
            print(f"{key}: {value}")
        out_string.append(value)
    return ",".join(out_string)
import json

def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {file_path}: {e}")
        return None

def store_json(file_path, data):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        print(f"Data successfully stored in {file_path}.")
    except Exception as e:
        print(f"Error storing data in {file_path}: {e}")


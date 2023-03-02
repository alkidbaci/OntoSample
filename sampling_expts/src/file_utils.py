import json
import random
import csv
import tempfile
import os
from typing import List, Dict, Any

from pathlib import Path

def create_temp_file(suffix=".tsv") -> str:
    return tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name

def write_jsonl_or_json(path, data_points, shuffle=False, append=False):
    ensure_path(path, is_dir=False)
    with open(path, 'w' if not append else 'a') as f:
        if not data_points:
            f.write("")
        elif path.endswith('.json'):
            json.dump(data_points, f, indent=4)
        elif path.endswith('.jsonl'):
            if shuffle:
                random.shuffle(data_points)
            for item in data_points:
                json.dump(item, f)
                f.write('\n')

def read_jsonl_or_json_or_tsv(path, default_headers=[]):
    return read_jsonl_or_json(path) if path.endswith(".json") or path.endswith(".jsonl") else tsv_to_json_entries(path, default_headers)

def read_tsv(file_name):
    with open(file_name, 'r') as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        rows = []
        for row in rd:
            rows.append(row)
    return rows

def write_tsv(path, tsv_content):
    pass

def write_txt(path, data_points):
    with open(path, 'w') as outfile:
        for d in data_points:
            outfile.write(f"{d}\n")

def write_tsv_from_dict(path, data_points, headers):
    with open(path, 'w') as outfile:
        for d in data_points:
            outfile.write("\t".join([d[x] for x in headers]))
            outfile.write("\n")

def write_jsonl_or_json_or_tsv(path, data_points_dict_arr, headers_arr_for_tsv):
    return write_jsonl_or_json(path, data_points_dict_arr) if path.endswith(".json") or path.endswith(".jsonl") else write_tsv_from_dict(path=path, data_points=data_points_dict_arr, headers=headers_arr_for_tsv)

def read_jsonl_or_json(path):
    if not os.path.exists(path):
        raise Exception('File expected at ' + path + ' not found')
    records = []
    with open(path)  as f:
        if path.endswith('.json'):
            records = json.load(f)
        elif path.endswith('.jsonl'):
            records = [json.loads(l) for l in f]
    return records

def json_to_tsv(json_or_jsonl_path, get_fields_of_interest=None, sep="\t"):
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix = '.tsv').name
    with open(out_path, 'w') as outf:
        if get_fields_of_interest is not None:
            outf.write(f"{sep.join(get_fields_of_interest(None))}\n")
            for x in read_jsonl_or_json(json_or_jsonl_path):
                outf.write(f"{sep.join(get_fields_of_interest(x))}\n")
    print(f"\nJson ({json_or_jsonl_path}) -> TSV ({out_path})\n")
    return out_path

def tsv_to_json_entries(path, default_headers, delimiter='\t') -> List[Dict[str, Any]]:
    json_entries = []
    with open(path) as infile:
        if default_headers is not None and len(default_headers) > 0: # suppose the tsv has no header then specify header arr.
            reader = csv.DictReader(infile, delimiter=delimiter, fieldnames=default_headers)
        else:  # assume that the first line in the tsv is the header.
            reader = csv.DictReader(infile, delimiter=delimiter)
        for d in reader:
            json_entries.append(json.loads(json.dumps(d)))
    return json_entries

def ensure_path(fp: str, is_dir: bool):
    # Create parent level subdirectories if not exists.
    p = Path(fp)
    if os.path.exists(fp):
        return
    # if path indicates a dir: Create parent level subdirectories if not exists.
    if not is_dir and not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    # if path indicates a file: Create parent level subdirectories if not exists
    elif not p.exists() and is_dir:
        p.mkdir(parents=True, exist_ok=True)


def write_list_to_file(list_data:List, fp):
    with open(fp,'w') as outfile:
        for d in list_data:
            outfile.write(f"{d.strip()}\n")



import hashlib
import os
import argparse
from typing import Dict

from tqdm import tqdm

def calculate_md5(file_path: str, chunk_size: int = 8192) -> str:
    '''
    计算单个文件的 MD5 摘要，带有阅后即焚的进度条。
    '''
    md5_hash = hashlib.md5()
    try:
        total_size = os.path.getsize(file_path)
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, 
                  desc=f'Hashing {os.path.basename(file_path)}', leave=False) as pbar:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(chunk_size), b''):
                    md5_hash.update(chunk)
                    pbar.update(len(chunk))
                    
        return md5_hash.hexdigest()
    except Exception as e:
        return f'Error: {e}'

def hash_target(target_path: str) -> Dict[str, str]:
    results = {}
    if os.path.isfile(target_path):
        results[target_path] = calculate_md5(target_path)
    elif os.path.isdir(target_path):
        for root, _, files in os.walk(target_path):
            for file in files:
                file_path = os.path.join(root, file)
                results[file_path] = calculate_md5(file_path)
    else:
        raise ValueError(f'无效的路径: {target_path}')
    
    return results

def main():
    parser = argparse.ArgumentParser(description='计算文件或目录下所有文件的 MD5 值')
    parser.add_argument('target', type=str, help='需要计算 MD5 的文件或文件夹路径')
    args = parser.parse_args()

    try:
        results = hash_target(args.target)
        for path, md5_val in results.items():
            path = path.replace('\\', '/')
            print(f'{md5_val} ➜ {path}')
    except ValueError as e:
        print(e)

if __name__ == '__main__':
    main()
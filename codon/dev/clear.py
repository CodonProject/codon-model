import os
import shutil
import argparse

def clear_pycache(target_dir: str):
    '''
    递归遍历并删除目标路径下的所有 __pycache__ 文件夹。
    
    参数:
        target_dir (str): 需要清理的根目录路径。
    '''
    deleted_count = 0
    target_dir = os.path.abspath(target_dir)

    print(f'开始清理 {target_dir} 下的 __pycache__ ...')
    
    for root, dirs, files in os.walk(target_dir):
        if '__pycache__' in dirs:
            cache_dir = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(cache_dir)
                print(f'Deleted: {cache_dir}')
                deleted_count += 1
                
                dirs.remove('__pycache__')
            except Exception as e:
                print(f'Error deleting {cache_dir}: {e}')
                
    print(f'清理完成。共移除了 {deleted_count} 个 __pycache__ 缓存目录。')

def main():
    parser = argparse.ArgumentParser(description='递归清理目标路径下的所有 __pycache__ 文件夹')
    parser.add_argument('target_path', type=str, nargs='?', default='.', 
                        help='要清理的根目录路径 (默认为当前所在目录)')
    args = parser.parse_args()

    if not os.path.isdir(args.target_path):
        print(f'错误: 目录不存在 - {args.target_path}')
        return

    clear_pycache(args.target_path)
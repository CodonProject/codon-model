import argparse
import os

from codon.dev.hash import hash_target
from codon.dev.clear import clear_pycache

def handle_hash(args):
    '''处理 codon hash 命令'''
    try:
        results = hash_target(args.filename_or_dirname)
        for path, md5_val in results.items():
            print(f'{md5_val}  {path}')
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f'发生意外错误: {e}')

def handle_clear(args):
    '''处理 codon clear 命令'''
    if not os.path.isdir(args.target_path):
        print(f'错误: 目录不存在 - {args.target_path}')
        return
    clear_pycache(args.target_path)

def main():
    parser = argparse.ArgumentParser(
        prog='codon',
        description='Codon Model 统一命令行工具箱'
    )
    
    subparsers = parser.add_subparsers(title='可用子命令', dest='command', required=True)

    # codon hash
    parser_hash = subparsers.add_parser('hash', help='计算文件或目录的 MD5 值')
    parser_hash.add_argument('filename_or_dirname', type=str, help='需要计算 MD5 的文件或文件夹路径')
    parser_hash.set_defaults(func=handle_hash)

    # codon clear
    parser_clear = subparsers.add_parser('clear', help='递归清理目录下的所有 __pycache__ 缓存')
    parser_clear.add_argument('target_path', type=str, nargs='?', default='.', 
                              help='要清理的根目录路径 (默认为当前所在目录)')
    parser_clear.set_defaults(func=handle_clear)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
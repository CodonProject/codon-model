from install import clean, build
import os
import shutil
import subprocess

if __name__ == '__main__':
    clean()
    build()
    dirs_to_remove = ['build']
    for root, dirs, _ in os.walk('.'):
        for name in dirs:
            if name.endswith('.egg-info'):
                dirs_to_remove.append(os.path.join(root, name))
    
    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            print(f'Removing {dir_path}...')
            shutil.rmtree(dir_path)
    try:
        subprocess.run(['codon', 'clear'], check=True)
    except: pass
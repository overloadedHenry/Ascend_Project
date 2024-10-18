import os
import sys
from pathlib import Path
path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
path = path / 'filestore'

def save_files(file=None, path=path):
    os.chdir(path)
    if not os.path.exists('pdf'):
        os.makedirs('pdf')
    if not os.path.exists('excel'):
        os.makedirs('excle')
    if not os.path.exists('word'):
        os.makedirs('word') 

        



if __name__ == '__main__':
    
   save_files()
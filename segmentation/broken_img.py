import os
import shutil
import argparse
import numpy as np
from PIL import Image
from utils import create_directory

def main(from_path, to_path):
    create_directory(to_path)
    i=1
    for (r, dirs, files) in os.walk(from_path):
        for f in filter(lambda f: f[-3:]=="jpg", files):
            try:
                image = np.asarray(Image.open(os.path.join(r, f)))
                del image
            except OSError:
                shutil.move(os.path.join(r, f), os.path.join(to_path, f))
                print(f'move {os.path.join(r, f)}')
                if os.path.exists(os.path.join(r, f.replace('jpg', 'json'))):
                    shutil.move(os.path.join(r, f.replace('jpg', 'json')), os.path.join(to_path, f.replace('jpg', 'json')))
                    print(f"move {os.path.join(r, f.replace('jpg', 'json'))}")
                else:
                    print(f"{os.path.join(r, f.replace('jpg', 'json'))} does not exit.")
            if i%200 == 0:
                print(f'{i}')
            
            i+=1
                                  
    

            
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--from-path', default='../../data/A2_비듬_각질_상피성잔고리', type=str, help='')
    parser.add_argument('--to-path', default='../../data/A2_비듬_각질_상피성잔고리_broken', type=str, help='')
    args=parser.parse_args()
                  
    main(args.from_path, args.to_path)
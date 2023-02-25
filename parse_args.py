import json 
import os

def read_args():
    args_path = os.environ['ARGSPATH']

    with open(args_path) as json_args:
        args_dict = dict(json.load(json_args))
        return args_dict
    
def create_params():
    args_dict = read_args()
    arguments = [f'--{k}={v}' for k, v in args_dict]
    return ' '.join(arguments)

if __name__ == '__main__':
     print(create_params())
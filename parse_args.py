import json 
import os
from io_utils import create_parser

def read_args():
    args_path = os.environ.get('ARGSPATH')

    with open(args_path) as json_args:
        args_dict = dict(json.load(json_args))
        return args_dict
    
def param_form(k, v, parser):
    if type(v) == bool and v:
        return f'--{k}'
    
    if not v or str(parser.get_default(v)) == str(v):
        return ''
    
    if k == "checkpoint_suffix":
        return f'--{k} "{v}"'
    
    return f'--{k} {v}'
    
def create_params():
    args_dict = read_args()
    parser = create_parser('train')
    arguments = [param_form(k,v, parser) for k, v in args_dict.items()]
    return ' '.join(arguments)

if __name__ == '__main__':
     print(create_params())
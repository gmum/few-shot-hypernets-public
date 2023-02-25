import json 
import os
from io_utils import create_parser

def read_args():
    args_path = os.environ.get('ARGSPATH')

    with open(args_path) as json_args:
        args_dict = dict(json.load(json_args))
        return args_dict
    
def param_form(k, v, parser):
    # Boolean actions are used without a value
    if type(v) == bool and v:
        return f'--{k}'
    
    # Skip default parameters or those that weren't used
    if not v or str(parser.get_default(k)) == str(v):
        return ''
    
    if k == "checkpoint_suffix":
        # This just helps bash to properly parse params and does not have any impact on the model. 
        # Also, it indicates that we perform an experiment
        new_v = "EXPERIMENT_" + v.replace(' ', '_') 
        return f'--{k} "{new_v}"'
    
    return f'--{k} {v}'
    
def create_params():
    args_dict = read_args()
    parser = create_parser('train') # We need it to get default values of the parser
    arguments = [param_form(k,v, parser) for k, v in args_dict.items()]
    return ' '.join(arguments)

if __name__ == '__main__':
     print(create_params())
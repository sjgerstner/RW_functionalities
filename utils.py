from argparse import Namespace
import os

def get_data_dir(args:Namespace):
    if args.data_dir is None:
        if "WORK" not in os.environ:
            os.environ["WORK"]='..'
        data_dir = os.environ["WORK"] + '/RW_functionalities_results'
    else:
        data_dir = args.data_dir
    return data_dir

import argparse
from utils import *
from utils.config_utils import Nestedspace
def arg_parser():
    """
    @args:
        - No args
    @rets:
        - config (Namespace): runtime namespace for setup
    """
    parser = argparse.ArgumentParser("Argument parser for fastvdnet")
    parser.add_argument("--useLastImage", type=int, default=1, help='use the last image as the output')
    parser.add_argument("--time", type=int, default=5, help='get last n images from the series')
    parser.add_argument("--normalize_images", type=int, default=1, help='normalize the images 0-1')
    parser.add_argument("--model_type", type=str, default='FASTVDNET', help='type of model to use')
    parser.add_argument("--complex_i", type=int, default=0, help='use complex images')
    parser.add_argument("--usample", type=float, default=1.0, help='undersampling_factor')
    parser.add_argument("--no_in_channel", type=int, default=1, help='no of input channels')
    parser.add_argument("--batch_size", type=int, default=32, help='batch size')
    parser.add_argument("--train_files", type=str,nargs='+', default=['/Users/javeda2/Documents/data/_0_retro_cines_kspace_spiral.h5'], help='file names')
    parser.add_argument("--epochs", type=int, default=10, help='epochs')
    parser.add_argument("--cuda", type=int, default=1, help='using servers')
    parser.add_argument("--num_workers", type=int, default=1, help='num workers')
    parser.add_argument("--gpus", type=int, default=4, help='num workers')    
    parser.add_argument("--use_non_appended_keys", type=int, default=1, help='using servers')

    

    
    parser.add_argument("--exp_name", type=str, default="cuda_t3st", help='name of the experiment')
    ns = Nestedspace()
    
    args = parser.parse_args(namespace=ns)

    return args
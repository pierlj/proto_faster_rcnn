import sys
import os
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from .data.utils import is_notebook, build_sections_from_file
from .data.datasets_info import *

import torch

cuda = torch.cuda.is_available()
device = "cuda:0" if cuda else "cpu"
# device = "cpu"


"""Magic calling needed tqdm depending on environment"""

if is_notebook():
	mtqdm = tqdm_notebook
else:
	mtqdm = tqdm

class Config():
    '''
    Config object to regroup all parameters needed for training
    TO DO move this under ./train
    '''
    def __init__(self, args=None, logger=None):

        ### Dataset Choice
        self.DATASET_META = dota_meta
        # self.DATASET_META = mnist_meta

        ### Backbone
        self.PRETRAINED_BACKBONE = 'imagenet' #'./model_ssl_frozen_512.pth' # path to network from which backbone should be extracted
        self.BACKBONE_TRAINABLE_LAYERS = 3

        args = None # disable args 

        ### Training
        self.LR = args.lr if hasattr(args, 'lr') else 1e-4
        self.N_CLASSES = self.DATASET_META.n_classes + 1 
        self.SEED = args.seed if hasattr(args, 'seed') else 42
        self.N_EPOCH = args.epochs if hasattr(args, 'epochs') else 5000
        self.OPTIMIZER = args.optimizer if hasattr(args, 'optimizer') else 'Adam'
        self.SAVE_PATH = args.save_path if hasattr(args, 'save_path') else '/media/pierre/Data/ProtoFasterRCNN/Paper_exp'
        self.CHECKPOINT_PATH = args.checkpoint_path if hasattr(args, 'checkpoint_path') else None
        self.BATCH_SIZE = args.batch_size if hasattr(args, 'batch_size') else 1
        self.GAMMA = args.gamma if hasattr(args, 'gamma') else 0.5
        self.SCHEDULING_STEP = [20000, 40000]
        self.NOTIFY = args.notify if hasattr(args, 'notify') else False
        self.SPECTRAL_NORM = False
        self.FEW_SHOT = args.few_shot if hasattr(args, 'few_shot') else False
    
        ### Losses
        self.RPN_EMBEDDING_LOSS = ['triplet'] # 'triplet' / 'nll' / 'cos'
        self.RPN_HEM = True
        self.ROI_HEAD_EMBEDDING_LOSS = [] # 'triplet' / 'nll' / 'cos' 
        self.ROI_HEM = True
        self.KMEANS_NEG_CLUST = False
        self.KMEANS_GHOST_CLASSES = False


        ### Log and save
        self.TEST_INTERVAL = 500
        self.SAVE_INTERVAL = 2000
        self.LOG_INTERVAL = 5000
        self.LOG_EMBEDDINGS = False
        
        ### Debugging
        self.DEBUG = sys.gettrace() is not None
        self.OUTPUT_GRAD = False
        self.DEBUG_PLOT = sys.gettrace() is not None # for SSL
        

        ### Data augmentation
        self.DA_HFLIP = 0.5
        self.DA_VFLIP = 0.5
        self.DA_COLOR = 0.5
        self.DA_AFFINE = 0.5

        ### Few shot learning
        self.K_SHOTS = 1
        
        self.N_WAYS_TRAIN = 3
        self.N_WAYS_TEST = 3
        self.N_CLASSES_TEST = 3
        self.N_QUERY_TRAIN = 5
        self.N_QUERY_TEST = 5
        self.SPLIT_METHOD = 'deterministic' # 'random' / 'deterministic' / 'same' 

        ### Prototypes computation
        self.USE_REGULAR_RPN = False
        self.RPN_EMBEDDING_D = 128
        self.ROI_HEAD_EMBEDDING_D = 128
        self.RPN_SIGMA = 0.5
        self.RPN_SIGMA_MODE = 'fixed' # 'learned' / 'estimated' / 'decay' / 'fixed'
        self.ROI_HEAD_SIGMA = 0.5
        self.ROI_HEAD_SIGMA_MODE = 'fixed' # 'learned' / 'estimated' / 'decay' / 'fixed'
        self.EXACT_BOXES = True 
        self.RPN_PROTO_MODE = 'mean' # 'all' / 'mean'
        self.ROI_PROTO_MODE = 'mean' # 'all' / 'mean'
        self.RPN_POS_NEG_RATIO = 'all' # int or 'all'
        self.ROI_POS_NEG_RATIO = 3 # int or 'all
        self.RPN_EX_PER_CLASS = None
        self.ROI_EX_PER_CLASS = None
        self.MIN_BOX_SIZE = 0
        self.USE_PRIOR = False
        self.ALPHA = 1.0 # moving average mixing coef 0.0 keep randomly initialized proto, 1.0 change every step 

        ### SimCLR
        self.SSL = False
        self.SSL_PRETRAIN = self.SSL and True
        self.SSL_METHOD = 'simsiam' # 'vanilla', 'barlow'
        self.SSL_RANDOM_BOXES = False
        self.SSL_AUGMENT = True
        self.TEST_INTERVAL += 1e9 * int(self.SSL_PRETRAIN)
        self.SSL_IMG_SIZE = 512
        

        ### Postprocessing
        self.POST_PROCESSING_MODE = 'frcnn'
        self.BOX_NMS_THRESH = 0.8
        self.BOX_NMS_THRESH_CLS = 0.5
        self.BOX_SCORE_THRESH = 0.2
        self.SCORE_MIXING_LOC = 0.5
        self.SCORE_MIXING_CLS = 0.5
        self.DETECTION_PER_IMAGE = 100
        self.CLUST_BW = None

        ### Amp params
        self.OPT_AMP = 'O0'

        # Easier to keep writter in config object to use it anywhere
        self.logger = logger
        self.log_path = None
    
    def save(self):
        raise NotImplementedError
    
    def log_config(self):
        raise NotImplementedError

    def __str__(self):
        res = ''
        for attr in vars(self):
            res += '`{}`: `{}` \n'.format(attr, self.__getattribute__(attr))
        return res
    
    def to_dict(self, recursive=False):
        config_dict = {}
        for attr, value in self.__dict__.items():
            if type(value) in [int, float, str, bool, dict]:
                config_dict[attr] = value
            elif type(value) == DatasetMeta and recursive:
                config_dict[attr] = value.to_dict()
            else:
                config_dict[attr] = str(value)
        return config_dict
    
    def from_dict(self, config_dict):
        for key, value in config_dict.items():
            if key == 'DATASET_META':
                setattr(self, key, DatasetMeta(**value))
            else:
                setattr(self, key, value)
    

    def to_grouped_dict(self):
        config_dict = self.to_dict()
        config_sections = build_sections_from_file(__file__)
        grouped_config_dict = {}
        for section, params in config_sections.items():
            section_dict = {}
            for p in params:
                section_dict[p] = config_dict[p]
            grouped_config_dict[section] = section_dict

        return grouped_config_dict

class DummyConfig(Config):
    def __init__(self):
        empty_args = Args()
        super().__init__(empty_args)


class Args():
    def __init__(self):
        pass

class DummyArgs():
    def __init__(self, n_classes):
        self.lr = 1e-5
        self.n_classes = n_classes
        self.seed = 1
        self.epochs = 15
        self.optimizer = 'Adam'
        self.save_path = ''
        self.checkpoint_path = ''
        self.batch_size = 2
        self.gamma = 0.7
        self.notify = False
        self.DATASET_META = dota_meta

class NoStdStreams(object):
    def __init__(self,stdout = None, stderr = None):
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()

import torch
import collections
from .custom_model import CustomModelPrototypical, CustomModel
from ..config import *

def load_model(path, config=None, try_proto=True, load_pretrain_ssl=False, verbose=False):
    # load file from path as if it was a dict
    if path.endswith('pt'):
        model = path
    else:
        model_dict = torch.load(path)
        if config is None and type(model_dict) in [collections.OrderedDict, dict]:
            if 'config' in model_dict:
                config = Config(None)
                config.from_dict(model_dict['config'])
                print('Config loaded')
            else:
                config = Config()
                print('New config created')

        if try_proto and config is not None:
            try:
                # print('Try to load prototypical model from state_dict')
                model = CustomModelPrototypical(config)
                current_model_dict = model.state_dict()

                if 'state_dict' in model_dict:
                    model_weights = model_dict['state_dict']
                else:
                    model_weights = model_dict
                
                if load_pretrain_ssl:
                    filtered_weights = {k: v for k,v in model_weights.items() if k in current_model_dict}
                    current_model_dict.update(filtered_weights)
                    model.load_state_dict(current_model_dict)
                else:
                    model.load_state_dict(model_weights)
                print('Model loaded')
                return model
            except Exception as e:
                if verbose:
                    print(e)

        if config is not None:
            try:
                # print('Try to load  model from state_dict')
                model = CustomModel(config)
                if 'state_dict' in model_dict:
                    model.load_state_dict(model_dict['state_dict'])
                else:
                    model.load_state_dict(model_dict)
                print('Model loaded')
                return model
            except Exception as e:
                # if verbose:
                print(e)

        print('Model has been saved as a whole')
        model = model_dict
    return model


def save_model(model, config, path):
    torch.save({'state_dict':model.state_dict(),
            'config': config.to_dict(recursive=True)}, path)

def convert_model(path, config=None):
    if config is None:
        config = Config()
    model = load_model(path, config=config)
    save_model(model, config, path)
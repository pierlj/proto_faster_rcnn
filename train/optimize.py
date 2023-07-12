import os
import sys
import json
import argparse
import numpy as np
import optuna
from .custom_logger import CustomLogger
from .train import Trainer
from .train_episode import EpisodicTrainer
from ..config import Config
from ..data.dataset import ObjectDetectionDataset
from ..data.bbox import BBoxMode


class Callback(object):
    def __init__(self):
        self.callback_data = None

    def __call__(self, study, trial):
        #save model for instance
        self.callback_data = None

    def set_data(self, data):
        self.callback_data = data


class Objective(object):
    def __init__(self, callback, args, variables_to_search):
        self.callback = callback
        self.logger = CustomLogger(log_dir='phd_utils/runs/test', notify=args.notify)

        self.args = args
        self.variable_to_search = variables_to_search

    def __call__(self, trial):
        
        config = Config(self.args, logger=self.logger)
        if not config.DEBUG:
            config.log_path = self.logger.log_dir
            
        if self.variable_to_search != []:
            self.prepare_config_for_search(config, trial)
            
        if self.args.few_shot:
            self.trainer = EpisodicTrainer(
                logger=self.logger, trial=trial, config=config,
                n_ways_train=config.N_WAYS_TRAIN,
                n_ways_test=config.N_WAYS_TEST, k_shots=config.K_SHOTS,
                n_query_test=config.N_QUERY_TEST, n_query_train=config.N_QUERY_TRAIN)
        else:
            # dataset = ObjectDetectionDataset(
            #     'xView', '/home/pierre/Documents/Datasets/xView/train/prepared/',
            #     train_val_ratio=.997,
            #     bbox_modes=[BBoxMode.XYXY, BBoxMode.REL],
            #     config=config)

            dataset = ObjectDetectionDataset(
                config,
                train_val_ratio=.997,
                bbox_modes=[BBoxMode.XYXY, BBoxMode.REL],
                background_class=True)
            self.trainer = Trainer(self.logger, trial, config, dataset=dataset)

        try:
            loss = self.trainer.train()
        except Exception as err:
            loss = np.inf
            if sys.gettrace() is  None:
                self.logger.bot.report_error(err)
            raise err
        except KeyboardInterrupt:
            self.logger.bot.report_error('Training manually interrupted')
            sys.exit(0)

#         self.call_back.set_data(model.state_dict())# store model params in call back
        self.logger.writer.close()
        self.logger.new_run()
        return loss
    
    def prepare_config_for_search(self, cfg, trial):
        if 'lr' in self.variable_to_search:
            cfg.LR = trial.suggest_loguniform('lr', 1e-6, 1e-4)
        if 'rpn_hem' in self.variable_to_search:
            cfg.RPN_HEM = trial.suggest_categorical('RPN_HEM', [True, False])
        if 'roi_hem' in self.variable_to_search:
            cfg.ROI_HEM = trial.suggest_categorical('ROI_HEM', [True, False])
        if 'rpn_emb_loss' in self.variable_to_search:
            cfg.RPN_EMBEDDING_LOSS = trial.suggest_categorical('RPN_EMBEDDING_LOSS', ['triplet', 'nll', 'none'])
        if 'roi_emb_loss' in self.variable_to_search:
            cfg.ROI_HEAD_EMBEDDING_LOSS = trial.suggest_categorical('ROI_HEAD_EMBEDDING_LOSS', ['triplet', 'nll', 'none'])
        if 'rpn_sigma' in self.variable_to_search:
            cfg.RPN_SIGMA = trial.suggest_loguniform('RPN_SIGMA', 0.001,1.0)
        if 'roi_sigma' in self.variable_to_search:
            cfg.ROI_HEAD_SIGMA = trial.suggest_loguniform('ROI_HEAD_SIGMA', 0.001,1.0)
        if 'HEM' in self.variable_to_search:
            hem = trial.suggest_categorical('HEM', [True])
            cfg.RPN_HEM = hem
            cfg.ROI_HEM = hem
        if 'ALPHA' in self.variable_to_search:
            cfg.ALPHA = trial.suggest_categorical('ALPHA', [1.0])
        if 'KMEANS_GHOST' in self.variable_to_search:
            cfg.KMEANS_GHOST_CLASSES = trial.suggest_categorical('KMEANS_GHOST', [False])
        if 'KMEANS_CLUST' in self.variable_to_search:
            cfg.KMEANS_NEG_CLUST = trial.suggest_categorical('KMEANS_CLUST', [False])
        

        cfg.SAVE_PATH = os.path.join(cfg.SAVE_PATH, str(trial.params) + '_' + str(trial.datetime_start))
        if not os.path.isdir(cfg.SAVE_PATH):
            os.mkdir(cfg.SAVE_PATH)
            with open(os.path.join(cfg.SAVE_PATH, 'config.json'), 'w') as cfg_file:
                json.dump(cfg.to_dict(recursive=True), cfg_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detection training')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training')
    # parser.add_argument('--n-classes', type=int, default=11, metavar='N',
    #                     help='Number of classes for predictions')
    parser.add_argument('--epochs', type=int, default=25000, metavar='N',
                        help='number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.3, metavar='M',
                        help='Learning rate step gamma')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed')
    parser.add_argument('--save-path', type=str, default='model.pth',
                        help='For saving the current Model')
    parser.add_argument('--checkpoint-path', type=str, default='model_checkpoint.pt',
                        help='For saving the training checkpoint')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Optimizer to be used for training')
    parser.add_argument('--grid-search', type=str, default='',
                        help='Hyper parameters to optimize through optuna grid search, must be separated by \':\'')
    parser.add_argument('--n-trials', type=int, default=1,
                        help='Number of trials for grid search')
    parser.add_argument('--few-shot', action='store_true',
                        help='Enable FS learning')
    parser.add_argument('--notify', action='store_true',
                        help='Enable telegram bot notification')
    
    args = parser.parse_args()

    ALLOWED_GRID_SEARCH_VAR = ['HEM', 'ALPHA', 'KMEANS_GHOST', 'KMEANS_CLUST']
    n_trials = 1
    # variables_to_search = ['rpn_sigma', 'roi_sigma']
    # n_trials = 1
    # variables_to_search = ['HEM', 'ALPHA', 'KMEANS_GHOST', 'KMEANS_CLUST']
    variables_to_search = []

    if args.grid_search != '':
        variables_to_search = args.grid_search.split(':')
        variables_to_search = [var for var in variables_to_search if var in ALLOWED_GRID_SEARCH_VAR]
        n_trials = args.n_trials

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.NopPruner())
    callback = Callback()
    study.optimize(Objective(callback, args, variables_to_search), n_trials=n_trials, callbacks=[callback])

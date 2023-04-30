import time
import os
import logging
import sys
import numpy as np
import pandas as pd
import joblib as jl
import glob
import torch

from pivae_code import pi_vae, conv_pi_vae, datasets, util
from keras.callbacks import ModelCheckpoint
from numpy.random import Generator, PCG64
import cebra.datasets

import argparse

def _check_tensor(arr):
    if torch.is_tensor(arr):
        return arr.numpy()
    else:
        return np.array(arr)

def _to_batch_list(x, y, batch_size):
    if x is not None and y is not None:
        x = x.squeeze()
        if len(x.shape) == 3:
            x = x.transpose(0,2,1)
        x_batch_list = np.array_split(x, int(len(x) / batch_size))
        y_batch_list = np.array_split(y, int(len(y) / batch_size))
    else:
        return None, None
    return x_batch_list, y_batch_list

def make_loader(dataset, batch_size):
    x,u = _to_batch_list(_check_tensor(dataset[torch.arange(len(dataset))]), _check_tensor(dataset.index), batch_size)
    loader = pi_vae.custom_data_generator(x, u)
    _len = len(x)
    return x, u, loader, _len

class Task:

    def __init__(self, args):
        self.args = args
        self.get_data()
        self.model = self.get_model()
        if self.args.save_flag:
            self.get_logger()

    def get_model(self):
        pass

    def get_data(self):
        pass

    def get_logger(self):

        if not os.path.isdir(self.args.logdir):
            os.makedirs(self.args.logdir)

        self.log_name = os.path.join(self.args.logdir,
                                     'logger')
        self.log = logging.getLogger()
        for hdlr in self.log.handlers[:]:
            self.log.removeHandler(hdlr)
        formatter = logging.Formatter(
            "%(asctime)-15s::%(levelname)s::%(filename)s::%(funcName)s::%(lineno)d::%(message)s"
        )
        filehandler = logging.FileHandler(self.log_name + ".log", "a")
        filehandler.setFormatter(formatter)
        self.log.addHandler(filehandler)
        streamhandler = logging.StreamHandler(sys.stdout)
        streamhandler.setFormatter(formatter)
        self.log.addHandler(streamhandler)
        self.log.setLevel(logging.DEBUG)
        self.log.info(self.args)

    def optimize(self):
        if self.args.save_best_only:
            mcp = ModelCheckpoint(
                self.args.logdir+'/model_best.h5',
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
            )
        else:
            mcp = ModelCheckpoint(
                self.args.logdir+'/model_{epoch:07d}.h5',
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
            )
        fitting = self.model.fit_generator(
            self.train_loader,
            steps_per_epoch=self.len_train,
            epochs=self.args.epochs,
            verbose=1,
            validation_data=self.valid_loader,
            validation_steps=self.len_valid,
            callbacks=[mcp],
        )
        self.train_loss = fitting.history['loss']
        self.valid_loss = fitting.history['val_loss']
        
    def save_emission(self):
        _model = self.get_model()
        for ckp in glob.glob(f'{self.args.logdir}/*.h5'):            
            _model.load_weights(ckp)
            ckp_no = ckp.split('model_')[1].split('.h5')[0]
            emissions_with_label={}
            emissions_without_label = {}

            for (x, u, flag) in zip([self.x_train, self.x_valid, self.x_test], 
                                  [self.u_train, self.u_valid, self.u_test],
                                       ['train', 'valid', 'test']):
                if x is None or u is None:
                    emb_with_label = None
                    emb_without_label = None
                else:
                    emb_with_label = _model.predict([
                                np.concatenate(x),
                                np.concatenate(u)
                            ])[0]

                    emb_without_label = _model.predict([
                                    np.concatenate(x),
                                    np.concatenate(u)
                                ])[6]

                emissions_with_label[flag]=emb_with_label
                emissions_without_label[flag]=emb_without_label

            emissions_with_label['args'] = vars(self.args)
            emissions_without_label['args'] = vars(self.args)
            emissions_with_label['metrics'] = {'train_loss':self.train_loss, 'valid_loss':self.valid_loss}
            emissions_without_label['metrics'] = {'train_loss':self.train_loss, 'valid_loss':self.valid_loss}

            jl.dump(emissions_with_label, os.path.join(self.args.logdir, f'emissions_with_label_{ckp_no}.jl'))
            jl.dump(emissions_without_label, os.path.join(self.args.logdir, f'emissions_without_label_{ckp_no}.jl'))

class HippocampusCVTask(Task):
    
    def get_model(self):
        if self.args.time_window != 1:
            vae = conv_pi_vae.conv_vae_mdl(
                dim_x=self.dataset.neural.shape[1],
                dim_z=self.args.latent_dim,
                dim_u=self.args.label_dim,
                time_window=self.args.time_window,
                gen_nodes=60,
                n_blk=2,
                mdl="poisson",
                disc=False,
                learning_rate=self.args.lr,
            )
        else:
            vae = pi_vae.vae_mdl(
                dim_x=self.dataset.neural.shape[1],
                dim_z=self.args.latent_dim,
                dim_u=self.args.label_dim,
                gen_nodes=60,
                n_blk=2,
                mdl="poisson",
                disc=False,
                learning_rate=self.args.lr,
            )
        return vae
        
    def get_data(self):
        if self.args.time_window == 10:
            offset_right = 5
            offset_left = 5
        elif self.args.time_window == 1:
            offset_right = 1
            offset_left = 0
                
        def _call_dataset(offset_right, offset_left, split):
            dataset=cebra.datasets.init(self.args.data_path, split = split)
            dataset.offset = cebra.data.datatypes.Offset(offset_left,offset_right)
            return dataset
        
        all_set = _call_dataset(offset_right, offset_left, 'all')
        train_set=_call_dataset(offset_right, offset_left, 'train')
        valid_set=_call_dataset(offset_right, offset_left, 'valid')
        test_set=_call_dataset(offset_right, offset_left, 'test')

        self.dataset = all_set

        self.x_train, self.u_train, self.train_loader, self.len_train=make_loader(train_set, self.args.batch_size)
        self.x_valid, self.u_valid, self.valid_loader, self.len_valid=make_loader(valid_set, self.args.batch_size)
        self.x_test, self.u_test, self.test_loader, self.len_test=make_loader(test_set, self.args.batch_size)
        self.x_all, self.u_all, self.all_loader, self.len_all=make_loader(all_set, self.args.batch_size)

class SyntheticTask(HippocampusCVTask):
    
    def get_model(self):
        vae = pi_vae.vae_mdl(
            dim_x=self.dataset.neural.shape[1],
            dim_z=self.args.latent_dim,
            dim_u=self.args.label_dim,
            gen_nodes=60,
            n_blk=2,
            mdl="poisson",
            disc=False,
            learning_rate=self.args.lr,
        )
        return vae
    def get_data(self):
        offset_right = 1
        offset_left = 0
        def _call_dataset(offset_right, offset_left):
            dataset=cebra.datasets.init(self.args.data_path)
            dataset.offset = cebra.data.datatypes.Offset(offset_left,offset_right)
            return dataset
        
        all_set = _call_dataset(offset_right, offset_left)
        train_set=_call_dataset(offset_right, offset_left)
        valid_set=_call_dataset(offset_right, offset_left)
        
        train_set.split('train')
        valid_set.split('valid')
        self.dataset = all_set

        self.x_train, self.u_train, self.train_loader, self.len_train=make_loader(train_set, self.args.batch_size)
        self.x_valid, self.u_valid, self.valid_loader, self.len_valid=make_loader(valid_set, self.args.batch_size)
        self.x_all, self.u_all, self.all_loader, self.len_all=make_loader(all_set, self.args.batch_size)    
        self.x_test, self.u_test = (None, None)

class AllenCaTask(Task):

    def get_model(self):
        if self.args.time_window != 1:
            vae = conv_pi_vae.conv_vae_mdl(
                dim_x=self.dataset.neural.shape[1],
                dim_z=self.args.latent_dim,
                dim_u=self.dataset.index.shape[1],
                time_window=self.args.time_window,
                gen_nodes=60,
                n_blk=2,
                mdl="poisson",
                disc=False,
                learning_rate=self.args.lr,
            )
        else:
            vae = pi_vae.vae_mdl(
                dim_x=self.dataset.neural.shape[1],
                dim_z=self.args.latent_dim,
                dim_u=self.dataset.index.shape[1],
                gen_nodes=60,
                n_blk=2,
                mdl="poisson",
                disc=False,
                learning_rate=self.args.lr,
            )
        return vae

    def get_data(self):

        if self.args.time_window == 10:
            offset_right = 5
            offset_left = 5
        elif self.args.time_window == 1:
            offset_right = 1
            offset_left = 0

        def _call_dataset(offset_right, offset_left):
            dataset=cebra.datasets.init(self.args.data_path)
            dataset.offset = cebra.data.datatypes.Offset(offset_left,offset_right)
            return dataset

        all_set = _call_dataset(offset_right, offset_left)
        self.dataset = all_set
        self.x_all, self.u_all, self.all_loader, self.len_all=make_loader(all_set, self.args.batch_size)
        self.x_train, self.u_train, self.train_loader, self.len_train=make_loader(all_set, self.args.batch_size)
        self.x_valid, self.u_valid, self.valid_loader, self.len_valid=make_loader(all_set, self.args.batch_size)
        self.x_test, self.u_test = (None, None)

class MonkeyReachingActiveTask(Task):

    def get_model(self):
        if self.args.mode == 'continuous':
            disc = False
        elif self.args.mode == 'discrete':
            disc = True
        if self.args.time_window != 1:
            vae = conv_pi_vae.conv_vae_mdl(
                dim_x=self.dataset.neural.shape[1],
                dim_z=self.args.latent_dim,
                dim_u=self.args.label_dim,
                time_window=self.args.time_window,
                gen_nodes=60,
                n_blk=2,
                mdl="poisson",
                disc=disc,
                learning_rate=self.args.lr,
            )
        else:
            vae = pi_vae.vae_mdl(
                dim_x=self.self.dataset.neural.shape[1],
                dim_z=self.args.latent_dim,
                dim_u=self.args.label_dim,
                gen_nodes=60,
                n_blk=2,
                mdl="poisson",
                disc=disc,
                learning_rate=self.args.lr,
            )
        return vae

    @property
    def session(self):
        return 'active'

    def get_data(self):

        if self.args.time_window == 10:
            offset_right = 5
            offset_left = 5
        elif self.args.time_window == 1:
            offset_right = 1
            offset_left = 0

        def _call_dataset(offset_right, offset_left, split):
            dataset=cebra.datasets.init(self.args.data_path)
            dataset.offset = cebra.data.datatypes.Offset(offset_left,offset_right)
            dataset.split(split)
            if self.args.mode == 'continuous':
                dataset.index = dataset.continuous_index
            elif self.args.mode == 'discrete':
                dataset.index = dataset.discrete_index
            return dataset

        all_set = _call_dataset(offset_right, offset_left, 'all')
        train_set = _call_dataset(offset_right, offset_left, 'train')
        valid_set = _call_dataset(offset_right, offset_left, 'valid')
        test_set = _call_dataset(offset_right, offset_left, 'test')
        self.dataset = all_set
        self.x_train, self.u_train, self.train_loader, self.len_train=make_loader(train_set, self.args.batch_size)
        self.x_valid, self.u_valid, self.valid_loader, self.len_valid=make_loader(valid_set, self.args.batch_size)
        self.x_test, self.u_test, self.test_loader, self.len_test=make_loader(test_set, self.args.batch_size)
        self.x_all, self.u_all, self.all_loader, self.len_all=make_loader(all_set, self.args.batch_size)



def main(args):

    if args.task == 'Synthetic':
        task = SyntheticTask(args)
    elif args.task == 'HippocampusCV':
        task = HippocampusCVTask(args)
    elif args.task == "AllenCa":
        task = AllenCaTask(args)
    elif args.task == "MonkeyReachingActive":
        task = MonkeyReachingActiveTask(args)

    if args.save_flag:
        task.get_logger()
    task.optimize()
    task.save_emission()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Training params
    parser.add_argument(
        "--data-path",
        type=str,
        default="/home/ubuntu/local/rat_hippocampus/achilles_data.mat",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--time-window", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save-flag", default=False, action="store_true")
    parser.add_argument("--save-best-only", default=False, action="store_true")
    parser.add_argument(
        "--logdir",
        type=str,
        default="pivae_test/0211/",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    #parser.add_argument("--split-no", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=100)

    ### Model params
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--label-dim", type=int, default=3)
    parser.add_argument("--task", type=str, default="AllenCa")
    parser.add_argument("--mode", type=str, default="continuous")
    parser.add_argument("--label-variable", type=str,
                        default='direction+position')  ## For monkey reaching data

    args = parser.parse_args()
    main(args)

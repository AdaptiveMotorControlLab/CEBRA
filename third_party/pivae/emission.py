import argparse
import json
import os
import glob
import re
import joblib as jl
import numpy as np
import sklearn

from examples.pi_vae import task


class piVAEEmission():
    def __init__(self, path,save=False):
        if not os.path.isdir(path):
            raise ValueError(f"{path} is not existing")
            
        
            
        def read_args(path):

            args = {}
            logfile = glob.glob(f'{path}/*/*.log')[0]
            with open(logfile, "r") as f:
                for line in f:
                    if "Namespace" in line:
                        p = line.strip(")\n").split("Namespace(")[-1]
                        p = re.split(r",\s", p)
                        args = {
                            arg.split("=")[0].strip(): eval(arg.split("=")[1])
                            for arg in p
                        }

            return args
        
        def knn_decode(emission, label):
            metric = 'euclidean'
            train_fs, train_label =  emission['train'], label['train']
            valid_fs, valid_label =  emission['valid'], label['valid']
            test_fs, test_label =  emission['test'], label['test']
            
            nn = np.power(np.linspace(1, 30, 5, dtype=int), 2)
            errs = []
            scs = []

            for n in nn:
                
                pos_knn = sklearn.neighbors.KNeighborsRegressor(n_neighbors=n,
                                                                metric=metric)
                dir_knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n,
                                                                 metric=metric)
                pos_knn.fit(train_fs, train_label[:,0])
                ## position is label[:,0], direction is label[:,1]
                dir_knn.fit(train_fs, train_label[:,1])
                pos_pred = pos_knn.predict(valid_fs)
                dir_pred = dir_knn.predict(valid_fs)
                pred = np.vstack([pos_pred, dir_pred]).T
                sc = sklearn.metrics.r2_score(valid_label[:, :2], pred)
                scs.append(sc)

            test_pos_knn = sklearn.neighbors.KNeighborsRegressor(
                n_neighbors=nn[np.argmax(scs)], metric=metric)
            test_dir_knn = sklearn.neighbors.KNeighborsClassifier(
                n_neighbors=nn[np.argmax(scs)], metric=metric)

            test_pos_knn.fit(np.concatenate([train_fs, valid_fs]), np.concatenate([train_label, valid_label])[:,0])
            test_dir_knn.fit(np.concatenate([train_fs, valid_fs]), np.concatenate([train_label, valid_label])[:,1])
            pos_pred = test_pos_knn.predict(test_fs)
            dir_pred = test_dir_knn.predict(test_fs)
            pred = np.vstack([pos_pred, dir_pred]).T
            test_score = sklearn.metrics.r2_score(test_label[:, :2], pred)
            pos_test_err = np.median(abs(pos_pred - test_label[:, 0]))
            pos_test_score = sklearn.metrics.r2_score(test_label[:, 0],
                                                      pos_pred)
            return [test_score, pos_test_score, pos_test_err], pred
            

        def get_emission(args):
            emission_w_label = {}
            emission_wo_label = {}
            label = {}
            args = argparse.Namespace(**args)
            args.save_flag = False
            model = glob.glob(self.path+ '/*/*.h5')[0]
            if args.task == 'Hippocampus':
                _task = task.HippocampusTask(args)
            elif args.task == 'HippocampusCV':
                _task = task.HippocampusCVTask(args)
            elif args.task == 'AllenCa':
                _task = task.AllenCaTask(args)
            elif args.task == 'MonkeyReachingActive':
                _task = task.MonkeyReachingActiveTask(args)

            _task.model.load_weights(model)
            for (x, u, flag) in zip([_task.x_all, _task.x_train, _task.x_valid, _task.x_test], 
                              [_task.u_all, _task.u_train, _task.u_valid, _task.u_test],
                                   ['all', 'train', 'valid', 'test']): 
                f_w_label = _task.model.predict([
                    np.concatenate(x),
                    np.concatenate(u)
                ])[0]


                f_wo_label = _task.model.predict([
                        np.concatenate(x),
                        np.concatenate(u)
                    ])[6]
                emission_w_label[flag]=f_w_label
                emission_wo_label[flag]=f_wo_label

                label[flag]=np.concatenate(u)
            return emission_w_label, emission_wo_label, label
        self.path = path
        args = read_args(self.path)
        self.emission_w_label, self.emission_wo_label, self.label=get_emission(args)
        decoding_wo_label_result, decoding_prediction = knn_decode(self.emission_wo_label, self.label)
        if save:
            jl.dump({'emission_w_label':self.emission_w_label,
                     'emission_wo_label': self.emission_wo_label,
                     'label': self.label}, f'{self.path}/emission.jl')
            jl.dump({'r2':decoding_wo_label_result[0],
                     'position_r2': decoding_wo_label_result[1],
                     'median_error': decoding_wo_label_result[2],
                     'prediction': decoding_prediction},
                    f'{self.path}/decoding_wo_label.jl')

def main(args):

    assert os.path.isdir(args.path), {'--path should be directory'}

    piVAEEmission(args.path, args.save)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='.', type=str)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    main(args=args)

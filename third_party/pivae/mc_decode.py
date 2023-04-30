import re
import os
import argparse
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import backend as K
import task
import joblib as jl
import sklearn

def read_args(logfile):
    args = {}
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


def compute_marginal_lik_poisson(vae_mdl,
                                 x_test,
                                 u_fake,
                                 n_sample,
                                 log_opt=False):
    lik_all = []

    for jj in range(len(x_test)):  ## for each batch
        lik_test = []
        for ii in range(len(u_fake)):  ## for each unique u value
            opts = vae_mdl.predict([x_test[jj], u_fake[ii][jj]])
            lam_mean = opts[4]
            lam_log_var = opts[5]
            z_dim = lam_mean.shape
            z_sample = np.random.normal(0,
                                        1,
                                        size=(n_sample, z_dim[0], z_dim[1]))
            z_sample = z_sample * np.exp(0.5 * lam_log_var) + lam_mean

            ## compute fire rate ##
            get_fire_rate_output = K.function(
                [vae_mdl.layers[-1].get_input_at(0)],
                [vae_mdl.layers[-1].get_output_at(0)],
            )
            fire_rate = get_fire_rate_output([z_sample.reshape(-1,
                                                               z_dim[-1])])[0]
            fire_rate = fire_rate.reshape(n_sample, -1, fire_rate.shape[-2],
                                          fire_rate.shape[-1])
            ## compute p(x|z) poisson likelihood ##
            loglik = x_test[jj] * np.log(np.clip(fire_rate, 1e-10,
                                                 1e7)) - fire_rate
            # n_sample*n_time*n_neuron
            loglik = loglik.sum(axis=(-2, -1), dtype = np.float64)
            ## sum across neurons
            loglik_max = loglik.max(axis=0)
            loglik -= loglik_max
            if log_opt:
                tmp = np.log(np.exp(loglik).mean(axis=0)) + (loglik_max)
            else:
                tmp = (np.exp(loglik).mean(axis=0)) * np.exp(loglik_max)
            lik_test.append(tmp)
        lik_all.append(np.array(lik_test))

    return lik_all


def decode_sampling_rat(test_x, test_y, model, sampling_num):
    hd_bins = np.linspace(0, 1.6, 100)
    hd_bins_dir = np.hstack([np.concatenate([np.linspace(0,1.6,100), np.linspace(0,1.6,100)])[...,None], np.zeros((200,2))]) 
    hd_bins_dir[:100][:,1]=1
    hd_bins_dir[100:200][:,2]=1
    nu_sample = 200
    u_fake = []
    for jj in range(nu_sample):
        tmp_all = []
        for ii in range(len(test_x)):
            nn = test_x[ii].shape[0]
            tmp = np.hstack((np.ones((nn, 1)) * hd_bins[jj % 100], np.zeros((nn, 2))))
            if jj >= (nu_sample // 2):
                tmp[:, 2] += 1
            else:
                tmp[:, 1] += 1
            tmp_all.append(tmp)
        u_fake.append(np.array(tmp_all))
    u_fake = np.array(u_fake)

    ## compute loglik

    lik_all = compute_marginal_lik_poisson(model, test_x, u_fake, sampling_num)
    decode_use = np.array(
        [
            (lik_all[jj]).reshape(200, -1, order="F").argmax(axis=0)
            for jj in range(len(lik_all))
        ]
    )
    median_err = np.median(
        np.abs(
            [
                hd_bins_dir[np.concatenate(decode_use)[i],0] - np.concatenate(test_y)[i, 0]
                for i in range(len(np.concatenate(test_y)))
            ]
        )
    )

    prediction = [
        hd_bins_dir[np.concatenate(decode_use)[i]]
        for i in range(len(np.concatenate(test_y)))
    ]

    return median_err, prediction

def decode_sampling_monkey_target(test_x, test_y, model, sampling_num):
    ## sample u
    u_fake = np.array(
        [[np.ones((test_x[ii].shape[0], 1)) * jj
          for ii in range(len(test_x))]
         for jj in range(8)])
    

    ## compute loglik
    lik_all = compute_marginal_lik_poisson(model,
                                           test_x,
                                           u_fake,
                                           sampling_num,
                                           log_opt=True)
    decode_use = np.array(
        [lik_all[jj].argmax(axis=0) for jj in range(len(lik_all))])

    acc = 1 - np.concatenate([
        ((test_y[jj]) != (decode_use[jj])) for jj in range(len(test_y))
    ]).mean()
    
    pred = np.concatenate([decode_use[jj] for jj in range(len(test_y))])
    true = np.concatenate([test_y[jj] for jj in range(len(test_y))])

    return acc, pred


def decode_sampling_monkey_pos(test_x, test_y, model, sampling_num):
    ## sample u
    hd_bin = np.array([[x_pos, y_pos]
          for x_pos in np.arange(-13, 13, 1)
         for y_pos in np.arange(-13, 13, 1)])
    u_fake = []
    for b in hd_bin:
        tmp = []
        for i in range(len(test_x)):
            tmp.append([b]*len(test_x[i]))
        u_fake.append(tmp)
    u_fake = np.array(u_fake)
    ## compute loglik
    lik_all = compute_marginal_lik_poisson(model,
                                           test_x,
                                           u_fake,
                                           sampling_num,
                                           log_opt=True)
    
    predictions=hd_bin[np.concatenate(lik_all, axis=1).argmax(axis=0)]
   
    return sklearn.metrics.r2_score(predictions, np.concatenate(test_y)), predictions


def decode(path, sampling_num):
    
    logfile = glob.glob(f'{path}/*.log')[0]
    args = read_args(logfile)
    args = argparse.Namespace(**args)
    args.save_flag = False
    models_list = glob.glob(path + "/*.h5")
    models_list.sort()
    model=models_list[-1]
    task_ = eval(f'task.{args.task}Task')(
        args)  
    task_.model.load_weights(model)
    if 'Hippocampus' in args.task:
        pos_error, prediction = decode_sampling_rat(task_.x_test,
                                                    task_.u_test,
                                                    task_.model, sampling_num)
        r2_score = sklearn.metrics.r2_score(np.concatenate(task_.u_test),
                                            np.array(prediction))
        pos_r2_score = sklearn.metrics.r2_score(
            np.concatenate(task_.u_test)[:, 0],
            np.array(prediction)[:, 0])
        decoding_results = {'r2': r2_score, 'position_r2': pos_r2_score, 'median_err': pos_error}
        
    elif 'MonkeyReachingActive' in args.task:
        if args.mode == 'continuous':
            r2_score, prediction=decode_sampling_monkey_pos(task_.x_test, task_.u_test, task_.model, sampling_num)
            decoding_results = {'r2': r2_score}
        
        elif args.mode =='discrete':
            acc, prediction=decode_sampling_monkey_target(task_.x_test, task_.u_test, task_.model, sampling_num)
            decoding_results = {'acc': acc}
        
        
    else:
        raise Exception('This is function for decoding on rat hippocampus data')

    return decoding_results, prediction, path



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=".", type=str)
    parser.add_argument("--savepath", default=".", type=str)
    parser.add_argument("--sampling-num", default=100, type=int)
    args = parser.parse_args()
    decoding_results, prediction, model_path = decode(args.path, args.sampling_num)
    jl.dump({'decoding_results':decoding_results, 'prediction': prediction}, os.path.join(model_path, 'pivae_mc_decoding.jl'))
    

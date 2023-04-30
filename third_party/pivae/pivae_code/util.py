import numpy as np
import scipy.special as ssp


def get_tc(y, hd, bin_len=40):  # compute empirical tunning curve of data
    hd_bins = np.linspace(hd.min(), hd.max(), bin_len)
    tuning_curve = np.zeros((len(hd_bins) - 1, y.shape[1]))
    for ii in range(len(hd_bins) - 1):
        data_pos = ((hd >= hd_bins[ii]) * (hd <= hd_bins[ii + 1]))
        tuning_curve[ii, :] = y[data_pos, :].mean(axis=0)
    return tuning_curve


def get_disc_tc(y, hd):  # compute empirical tunning curve of data
    hd_bins = np.unique(hd)
    tuning_curve = np.zeros((len(hd_bins), y.shape[1]))
    for ii in range(len(hd_bins)):
        data_pos = (hd == hd_bins[ii])
        tuning_curve[ii, :] = y[data_pos, :].mean(axis=0)
    return tuning_curve


def get_var(y, hd, bin_len=40):  # compute empirical tunning curve of data
    hd_bins = np.linspace(hd.min(), hd.max(), bin_len)
    tuning_curve = np.zeros((len(hd_bins) - 1, y.shape[1]))
    for ii in range(len(hd_bins) - 1):
        data_pos = ((hd >= hd_bins[ii]) * (hd <= hd_bins[ii + 1]))
        tuning_curve[ii, :] = y[data_pos, :].var(axis=0)
    return tuning_curve


def rolling_mean(x, N=10):
    return np.convolve(x, np.ones(N) / N, 'valid')


from keras import backend as K


## sample from p(z|u)
def compute_marginal_lik_poisson(vae_mdl,
                                 y_test,
                                 u_fake,
                                 n_sample,
                                 log_opt=False):
    lik_all = []
    for jj in range(len(y_test)):  ## for each batch
        lik_test = []
        for ii in range(len(u_fake)):  ## for each unique u value
            opts = vae_mdl.predict([y_test[jj], u_fake[ii][jj]])
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
                [vae_mdl.layers[-1].get_output_at(0)])
            fire_rate = get_fire_rate_output([z_sample.reshape(-1,
                                                               z_dim[-1])])[0]
            fire_rate = fire_rate.reshape(n_sample, -1, fire_rate.shape[-1])

            ## compute p(x|z) poisson likelihood ##
            loglik = y_test[jj] * np.log(np.clip(fire_rate, 1e-10,
                                                 1e7)) - fire_rate
            # n_sample*n_time*n_neuron
            loglik = loglik.sum(axis=-1)
            ## sum across neurons
            loglik_max = loglik.max(axis=0)
            loglik -= loglik_max
            if log_opt:
                tmp = np.log(np.exp(loglik).mean(axis=0)) + (loglik_max)
            else:
                tmp = (np.exp(loglik).mean(axis=0)) * np.exp(loglik_max)
            lik_test.append(tmp)
        lik_all.append(np.array(lik_test))
    #loglik_all = np.array(loglik_all);
    return lik_all


def get_min_max(x, y):
    return min(x.min(), y.min()), max(x.max(), y.max())


def get_min_max_3(x, y, z):
    return min(x.min(), y.min(), z.min()), max(x.max(), y.max(), z.max())

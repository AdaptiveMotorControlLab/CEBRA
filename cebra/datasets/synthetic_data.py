#
# (c) All rights reserved. ECOLE POLYTECHNIQUE FÉDÉRALE DE LAUSANNE,
# Switzerland, Laboratory of Prof. Mackenzie W. Mathis (UPMWMATHIS) and
# original authors: Steffen Schneider, Jin H Lee, Mackenzie W Mathis. 2023.
#
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/LICENSE.md
#

import os

import joblib

import cebra.data
from cebra.datasets import get_datapath
from cebra.datasets import parametrize

_DEFAULT_DATADIR = get_datapath()

synthetic_data_urls = {
    "continuous_label_refractory_poisson": {
        "url":
            "https://figshare.com/ndownloader/files/41668815?private_link=7439c5302e99db36eebb",
        "checksum":
            "fcd92bd283c528d5294093190f55ceba"
    },
    "continuous_label_t": {
        "url":
            "https://figshare.com/ndownloader/files/41668818?private_link=7439c5302e99db36eebb",
        "checksum":
            "a6e76f274da571568fd2a4bf4cf48b66"
    },
    "continuous_label_uniform": {
        "url":
            "https://figshare.com/ndownloader/files/41668821?private_link=7439c5302e99db36eebb",
        "checksum":
            "e67400e77ac009e8c9bc958aa5151973"
    },
    "continuous_label_laplace": {
        "url":
            "https://figshare.com/ndownloader/files/41668824?private_link=7439c5302e99db36eebb",
        "checksum":
            "41d7ce4ce8901ae7a5136605ac3f5ffb"
    },
    "continuous_label_poisson": {
        "url":
            "https://figshare.com/ndownloader/files/41668827?private_link=7439c5302e99db36eebb",
        "checksum":
            "a789828f9cca5f3faf36d62ebc4cc8a1"
    },
    "continuous_label_gaussian": {
        "url":
            "https://figshare.com/ndownloader/files/41668830?private_link=7439c5302e99db36eebb",
        "checksum":
            "18d66a2020923e2cd67d2264d20890aa"
    },
    "continuous_poisson_gaussian_noise": {
        "url":
            "https://figshare.com/ndownloader/files/41668833?private_link=7439c5302e99db36eebb",
        "checksum":
            "1a51461820c24a5bcaddaff3991f0ebe"
    },
    "sim_100d_poisson_cont_label": {
        "url":
            "https://figshare.com/ndownloader/files/41668836?private_link=7439c5302e99db36eebb",
        "checksum":
            "306b9c646e7b76a52cfd828612d700cb"
    }
}


@parametrize(
    "continuous-label-{name}",
    name=["t", "uniform", "laplace", "poisson", "gaussian"],
)
class SyntheticData(cebra.data.SingleSessionDataset):
    """
    Synthetic datasets with poisson, gaussian, laplace, uniform,
    and t noise during generative process.
    """

    def __init__(self, name, root=_DEFAULT_DATADIR, download=True):

        name = f"continuous_label_{name}"
        location = os.path.join(root, "synthetic")
        file_path = os.path.join(location, f"{name}.jl")

        super().__init__(download=download,
                         data_url=synthetic_data_urls[name]["url"],
                         data_checksum=synthetic_data_urls[name]["checksum"],
                         location=location,
                         file_name=f"{name}.jl")

        data = joblib.load(file_path)
        self.data = data  #NOTE: making it backwards compatible with synth notebook.
        self.name = name
        self.neural = self.data['z']
        self.latents = self.data['x']
        self.u = self.data['u']
        self.lam = self.data['lam']

    @property
    def input_dimension(self):
        return self.neural.size(1)

    @property
    def continuous_index(self):
        return self.index

    def __getitem__(self, index):
        """Return [ No.Samples x Neurons x 10 ]"""
        index = self.expand_index(index)
        return self.neural[index].transpose(2, 1)

    def __len__(self):
        return len(self.neural)

    def __repr__(self):
        return f"SyntheticData(name: {self.name}, shape: {self.neural.shape})"

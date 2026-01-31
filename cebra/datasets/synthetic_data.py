#
# CEBRA: Consistent EmBeddings of high-dimensional Recordings using Auxiliary variables
# © Mackenzie W. Mathis & Steffen Schneider (v0.4.0+)
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/blob/main/LICENSE.md
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
            "https://cebra.fra1.digitaloceanspaces.com/data/synthetic/continuous_label_refractory_poisson.jl.gz",
        "checksum":
            "fcd92bd283c528d5294093190f55ceba",
        "gzipped_checksum":
            "3641eed973b9cae972493c70b364e981"
    },
    "continuous_label_t": {
        "url":
            "https://cebra.fra1.digitaloceanspaces.com/data/synthetic/continuous_label_t.jl.gz",
        "checksum":
            "a6e76f274da571568fd2a4bf4cf48b66",
        "gzipped_checksum":
            "1dc8805e8f0836c7c99e864100a65bff"
    },
    "continuous_label_uniform": {
        "url":
            "https://cebra.fra1.digitaloceanspaces.com/data/synthetic/continuous_label_uniform.jl.gz",
        "checksum":
            "e67400e77ac009e8c9bc958aa5151973",
        "gzipped_checksum":
            "71d33bc56b89bc227da0990bf16e584b"
    },
    "continuous_label_laplace": {
        "url":
            "https://cebra.fra1.digitaloceanspaces.com/data/synthetic/continuous_label_laplace.jl.gz",
        "checksum":
            "41d7ce4ce8901ae7a5136605ac3f5ffb",
        "gzipped_checksum":
            "1563e4958031392d2b2e30cc4cd79b3f"
    },
    "continuous_label_poisson": {
        "url":
            "https://cebra.fra1.digitaloceanspaces.com/data/synthetic/continuous_label_poisson.jl.gz",
        "checksum":
            "a789828f9cca5f3faf36d62ebc4cc8a1",
        "gzipped_checksum":
            "7691304ee061e0bf1e9bb5f2bb6b20e7"
    },
    "continuous_label_gaussian": {
        "url":
            "https://cebra.fra1.digitaloceanspaces.com/data/synthetic/continuous_label_gaussian.jl.gz",
        "checksum":
            "18d66a2020923e2cd67d2264d20890aa",
        "gzipped_checksum":
            "0cb97a2c1eaa526e57d2248a333ea8e0"
    },
    "continuous_poisson_gaussian_noise": {
        "url":
            "https://cebra.fra1.digitaloceanspaces.com/data/synthetic/continuous_poisson_gaussian_noise.jl.gz",
        "checksum":
            "1a51461820c24a5bcaddaff3991f0ebe",
        "gzipped_checksum":
            "5aa6b6eadf2b733562864d5b67bc6b8d"
    },
    "sim_100d_poisson_cont_label": {
        "url":
            "https://cebra.fra1.digitaloceanspaces.com/data/synthetic/sim_100d_poisson_cont_label.npz.gz",
        "checksum":
            "306b9c646e7b76a52cfd828612d700cb",
        "gzipped_checksum":
            "768299435a167dedd57e29b1a6d5af63"
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

        super().__init__(
            download=download,
            data_url=synthetic_data_urls[name]["url"],
            data_checksum=synthetic_data_urls[name]["checksum"],
            gzipped_checksum=synthetic_data_urls[name].get("gzipped_checksum"),
            location=location,
            file_name=f"{name}.jl")

        data = joblib.load(file_path)
        self.data = data  #NOTE: making it backwards compatible with synth notebook.
        self.name = name
        self.neural = self.data['x']
        self.latents = self.data['z']
        self.index = self.data['u']
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

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
import argparse
import json
from dataclasses import MISSING
from typing import Literal, Optional

import literate_dataclasses as dataclasses

import cebra.data
import cebra.datasets


@dataclasses.dataclass
class Config:
    data: str = dataclasses.field(
        init=False,
        doc="""The dataset to run CEBRA on.
    Standard datasets are available in cebra.datasets.
    Your own datasets can be created by subclassing
    cebra.data.Dataset and registering the dataset
    using the ``@cebra.datasets.register`` decorator.
    """,
    )

    variant: str = dataclasses.field(
        default="single-session",
        doc="""The CEBRA variant to run.
    """,
    )

    logdir: str = dataclasses.field(
        default="/logs/single-rat-hippocampus-behavior/",
        doc="""Model log directory.
    This should be either a new empty
    directory, or a pre-existing directory containing a trained
    CEBRA model.
    """,
    )
    distance: str = dataclasses.field(
        default="cosine", doc="""Distance type to use in calculating loss""")

    loss_distance: str = dataclasses.field(
        default="cosine",
        doc=
        """Old version of 'distance' argument. Distance type to use in calculating loss""",
    )

    temperature_mode: Literal["auto", "constant"] = dataclasses.field(
        default="constant",
        doc=
        """Temperature for InfoNCE loss. If 'auto', the temperature is learnable. If set to 'constant', it is fixed to the given value""",
    )

    temperature: float = dataclasses.field(
        default=1.0, doc="""Temperature for InfoNCE loss.""")

    min_temperature: Optional[float] = dataclasses.field(
        default=None, doc="""Minimum temperature for learnable temperature""")

    time_offset: int = dataclasses.field(
        default=10,
        doc=
        """ Distance (in time) between positive pairs. The interpretation of this parameter depends on
        the chosen conditional distribution, but generally a higher time offset increases the difficulty of
        the learning task, and (in a certain range) improves the quality of the representation. The time
        offset would typically be larger than the specified receptive field of the model.""",
    )

    delta: float = dataclasses.field(
        default=0.1,
        doc=
        """ Standard deviation of gaussian distribution if it is chossed to use 'delta' distribution.
        The positive sample will be chosen by closest sample to a reference which is sampled from the defined gaussian
        distribution.""",
    )

    conditional: str = dataclasses.field(
        default="time_delta",
        doc=
        """Type of conditional distribution. Valid standard methods are "time_delta", "time", "delta", and more
        methods can be added to the ``cebra.data`` registry.""",
    )

    num_steps: int = dataclasses.field(
        default=1000,
        doc="""Number of total training steps.
    Number of total training steps. Note that training duration
    of CEBRA is independent of the dataset size. The total training
    examples seen will amount to ``num-steps x batch-size``,
    irrespective of dataset size.
    """,
    )

    learning_rate: float = dataclasses.field(
        default=3e-4, doc="""Learning rate for Adam optimizer.""")
    model: str = dataclasses.field(
        default="offset10-model",
        doc=
        """Model architecture. Available options are 'offset10-model', 'offset5-model' and 'offset1-model'.""",
    )

    models: list = dataclasses.field(
        default_factory=lambda: [],
        doc=
        """Model architectures for multisession training. If not set, the model argument will be used for all sessions""",
    )

    batch_size: int = dataclasses.field(
        default=512, doc="""Total batch size for each training step.""")

    num_hidden_units: int = dataclasses.field(default=32,
                                              doc="""Number of hidden units.""")

    num_output: int = dataclasses.field(default=8,
                                        doc="""Dimension of output embedding""")

    device: str = dataclasses.field(
        default="cpu", doc="""Device for training. Options: cpu/cuda""")

    tqdm: bool = dataclasses.field(
        default=False, doc="""Activate tqdm for logging during the training""")

    save_frequency: int = dataclasses.field(
        default=None, doc="""Interval of saving intermediate model""")
    valid_frequency: int = dataclasses.field(
        default=100, doc="""Interval of validation in training""")

    train_ratio: float = dataclasses.field(
        default=0.8,
        doc="""Ratio of train dataset. The remaining will be used for
        valid and test split.""",
    )
    valid_ratio: float = dataclasses.field(
        default=0.1,
        doc="""Ratio of validation set after the train data split.
        The remaining will be test split""",
    )

    @classmethod
    def _add_arguments(cls, parser, **override_kwargs):
        _metavars = {int: "N", float: "val"}

        def _json(self):
            return json.dumps(self.__dict__)

        for field in dataclasses.fields(cls):
            if field.type == list:
                kwargs = dict(
                    metavar=field.default_factory(),
                    default=field.default_factory(),
                    help=f"{str(field.metadata['doc'])}",
                )
                kwargs.update(override_kwargs.get(field.name, {}))
                parser.add_argument("--" + field.name.replace("_", "-"),
                                    **kwargs,
                                    nargs="+")
            else:
                kwargs = dict(
                    type=field.type,
                    metavar=field.default,
                    default=field.default,
                    help=f"{str(field.metadata['doc'])}",
                )
                kwargs.update(override_kwargs.get(field.name, {}))
                parser.add_argument("--" + field.name.replace("_", "-"),
                                    **kwargs)

        return parser

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add arguments to the argument parser."""
        cls._add_arguments(
            parser,
            data={"choices": cebra.datasets.get_options()},
            device={"choices": ["cpu", "cuda"]},
        )
        return parser

    def asdict(self):
        return self.__dict__

    def as_namespace(self):
        return argparse.Namespace(**self.asdict())


def add_arguments(parser):
    """Add CEBRA command line arguments to an argparser."""
    return Config.add_arguments(parser)

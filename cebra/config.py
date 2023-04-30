import argparse
import json
from dataclasses import MISSING
from typing import Literal, Optional

import literate_dataclasses as dataclasses

import cebra.data
import cebra.datasets


@dataclasses.dataclass
class Config:
    Standard datasets are available in cebra.datasets.
    Your own datasets can be created by subclassing
    cebra.data.Dataset and registering the dataset
    using the ``@cebra.datasets.register`` decorator.
    )

    )

    This should be either a new empty
    directory, or a pre-existing directory containing a trained
    CEBRA model.
    )





        the chosen conditional distribution, but generally a higher time offset increases the difficulty of
        the learning task, and (in a certain range) improves the quality of the representation. The time

        """ Standard deviation of gaussian distribution if it is chossed to use 'delta' distribution.
        The positive sample will be chosen by closest sample to a reference which is sampled from the defined gaussian

        """Type of conditional distribution. Valid standard methods are "time_delta", "time", "delta", and more

    Number of total training steps. Note that training duration
    of CEBRA is independent of the dataset size. The total training
    examples seen will amount to ``num-steps x batch-size``,
    irrespective of dataset size.
    )

    )

    )







    @classmethod
    def _add_arguments(cls, parser, **override_kwargs):
        _metavars = {int: "N", float: "val"}

        def _json(self):
            return json.dumps(self.__dict__)

        for field in dataclasses.fields(cls):
            if field.type == list:
                kwargs.update(override_kwargs.get(field.name, {}))
                parser.add_argument("--" + field.name.replace("_", "-"),
                                    **kwargs,
                                    nargs="+")
            else:
                kwargs.update(override_kwargs.get(field.name, {}))
                parser.add_argument("--" + field.name.replace("_", "-"),
                                    **kwargs)

        return parser

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add arguments to the argument parser."""
        return parser

    def asdict(self):
        return self.__dict__

    def as_namespace(self):
        return argparse.Namespace(**self.asdict())


def add_arguments(parser):
    """Add CEBRA command line arguments to an argparser."""
    return Config.add_arguments(parser)

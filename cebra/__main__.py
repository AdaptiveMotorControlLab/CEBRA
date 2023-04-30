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
"""CEBRA command line interface.


"""

import argparse
import sys

import numpy as np
import torch

import cebra
import cebra.distributions as cebra_distr


def train(parser, kwargs):
    """Train a new CEBRA model, potentially starting from a checkpoint."""
    parser.add_argument("--variant", choices=cebra.CEBRA.get_variants())
    args, kwargs = parser.parse_known_args()
    cebra_cls = cebra.CEBRA.get_variant(args.variant)

    cebra_cls.add_arguments(parser)
    args, kwargs = parser.parse_known_args(kwargs)
    experiment = cebra_cls.from_args(args=args)

    parser.add_argument(
        "--override",
        "-r",
        action="store_true",
        help="Override an existing checkpoint (don't load).",
    )
    args, kwargs = parser.parse_known_args(kwargs)

    if not args.override:
        experiment.load()
    try:
        experiment.train()
    except KeyboardInterrupt:
        print("Training aborted. Saving the model.")
        sys.exit(1)
    except Exception as exception:
        print("Error occurred. Aborting training.")
        raise exception
    finally:
        experiment.save()


def transform(parser, kwargs):
    """Transform an existing dataset with a trained CEBRA model."""
    print("transform a dataset.")


def app(parser, kwargs):
    """Start server for serving the web interface of CEBRA."""
    from cebra.integrations.streamlit import App

    App.add_arguments(parser)
    args = parser.parse_args(kwargs)
    App.run(args)


def main():
    parser = argparse.ArgumentParser(
        "cebra", formatter_class=argparse.RawTextHelpFormatter, add_help=False)
    commands = {"train": train, "transform": transform, "app": app}
    parser.add_argument("--version", action="store_true")
    parser.add_argument(
        "command",
        choices=list(commands.keys()),
        help="The subcommand to run:\n" +
        "\n".join(f"{name}\t{cmd.__doc__}" for name, cmd in commands.items()),
        metavar="command",
    )
    args, kwargs = parser.parse_known_args()
    if args.version:
        print(f"CEBRA {cebra.__version__}.")
        sys.exit(0)
    command = commands.get(args.command)

    parser = argparse.ArgumentParser(f"cebra {args.command}")
    command(parser, kwargs)


if __name__ == "__main__":
    main()

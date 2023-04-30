from typing import Callable, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin

import cebra.data
import cebra.models
import cebra.solver


class CEBRA(BaseEstimator, TransformerMixin):

        verbose (bool):
        num_hidden_units (int):
        pad_before_transform (bool):
        hybrid (bool):
        optimizer_kwargs (dict):


    """

    @classmethod
        """Get a list of supported model architectures.

        These values can be directly passed to the ``model_architecture``
        argument.

        Args:
            pattern: Optional pattern for filtering the architecture list.
                Should use the :py:mod:`fnmatch` patterns.

        Returns:
            A list of all supported model architectures.

        Note:
            It is always possible to use the additional model architectures
            given by :py:func:`cebra.models.get_options` via the CEBRA pytorch
            API.
        """

        # TODO(stes): Check directly via the classes (but without initializing)
        return [
            option for option in cebra.models.get_options(pattern)
            if ("subsample" not in option and "resample" not in option and
                "supervised" not in option)
        ]

        optimizer: str = "adam",
        hybrid: bool = False,
        self.__dict__.update(locals())

        if self.optimizer != "adam":
            raise NotImplementedError(
                "Only adam optimizer supported currently.")



                raise ValueError(

    def _compute_offset(self) -> cebra.data.Offset:
        #            once a better solution is implemented in cebra.models
        return cebra.models.init(self.model_architecture,
                                 num_neurons=1,
                                 num_units=2,
                                 num_output=1).get_offset()


    def _prepare_criterion(self):
        if self.criterion == "infonce":
            if self.temperature_mode == "auto":
                if self.distance == "cosine":
                    return cebra.models.LearnableCosineInfoNCE(
                        temperature=self.temperature,
                elif self.distance == "euclidean":
                    return cebra.models.LearnableEuclideanInfoNCE(
                        temperature=self.temperature,
            elif self.temperature_mode == "constant":
                if self.distance == "cosine":
                    return cebra.models.FixedCosineInfoNCE(
                        temperature=self.temperature,)
                elif self.distance == "euclidean":
                    return cebra.models.FixedEuclideanInfoNCE(
                        temperature=self.temperature,)

        raise ValueError(f"Unknown similarity measure '{self.distance}' for "
                         f"criterion '{self.criterion}'.")


        Args:
            X: A 2D data matrix.
            y: An arbitrary amount of continuous indices passed as 2D matrices, and up to one
                discrete index passed as a 1D array. Each index has to match the length of ``X``.
            callback: If a function is passed here with signature ``callback(num_steps, solver)``,
            callback_frequency: Specify the number of iterations that need to pass before triggering

        Returns:
            ``self``, to allow chaining of operations.
        """
        return self
        """Transform an input sequence and return the embedding.
        Args:
            X: A numpy array or torch tensor of size ``time x dimension``.

        Returns:
        """


        input_dtype = X.dtype

        with torch.no_grad():
            if self.pad_before_transform:
                X = np.pad(X, ((offset.left, offset.right - 1), (0, 0)),
                           mode="edge")
            X = torch.from_numpy(X).float().to(self.device_)

                # Fully convolutional evaluation, switch (T, C) -> (1, C, T)
                X = X.transpose(1, 0).unsqueeze(0)
            else:
                # Standard evaluation, (T, C, dt)
            return output.astype(input_dtype)
        return output

    def fit_transform(
        self,
        *y,
        callback: Callable[[int, cebra.solver.Solver], None] = None,
        callback_frequency: int = None,
        self.fit(X,
                 *y,
                 callback=callback,
                 callback_frequency=callback_frequency)
        return self.transform(X)

    def _more_tags(self):
        # NOTE(stes): This tag is needed as seeding is not fully implemented in the
        # current version of CEBRA.

    def save(self, filename: str, backend: Literal["torch"] = "torch"):
        """Save the model to disk.

        Note:
            backward compatible yet between CEBRA versions!
        """
        if backend != "torch":
            raise NotImplementedError(f"Unsupported backend: {backend}")
        checkpoint = torch.save(self, filename)
        return checkpoint

    @classmethod
    def load(cls,
             filename: str,
             backend: Literal["torch"] = "torch",
             **kwargs) -> "CEBRA":
        """Load a model from disk.

            kwargs: Optional keyword arguments passed directly to the loader.
        Note:
            backward compatible yet between CEBRA versions!
        """
        if backend != "torch":
            raise NotImplementedError(f"Unsupported backend: {backend}")
        model = torch.load(filename, **kwargs)
        if not isinstance(model, cls):
            raise RuntimeError("Model loaded from file is not compatible with "
                               "the current CEBRA version.")
        return model

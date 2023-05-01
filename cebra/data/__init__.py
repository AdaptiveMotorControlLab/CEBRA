"""Data loaders use distributions and indices to make samples available for training.

CEBRA supports different dataset types out-of-the box:

- :py:class:`cebra.data.single_session.SingleSessionDataset` is the abstract base class for a single session dataset. Single session datasets
  have the same feature dimension across the samples (e.g., neural data) and all context
  variables (e.g. behavior, stimuli, etc.).
- :py:class:`cebra.data.multi_session.MultiSessionDataset` is the abstract base class for a multi session dataset.
  Multi session datasets contain of multiple single session datasets. Crucially, the dimensionality of the
  auxiliary variable dimension needs to match across the sessions, which allows alignment of multiple sessions.
  The dimensionality of the signal variable can vary arbitrarily between sessions.

package.

"""

# NOTE(stes): intentional ordering of imports to avoid circular imports
#             these imports will not be reordered by isort (see .isort.cfg)
from cebra.data.base import *
from cebra.data.datatypes import *

from cebra.data.single_session import *
from cebra.data.multi_session import *

from cebra.data.datasets import *

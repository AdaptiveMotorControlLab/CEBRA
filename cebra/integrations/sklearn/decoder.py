from typing import Generator, Optional, Tuple, Union
import numpy as np
import sklearn



        prediction = self.predict(X)
        test_score = sklearn.metrics.r2_score(y, prediction)





        return self


        for n in np.power(np.arange(1, 6, dtype=int), 2):





        for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:
            yield dict(alpha=alpha)

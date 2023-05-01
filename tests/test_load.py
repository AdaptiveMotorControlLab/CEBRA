
__test_functions = []

def _skip_hdf5storage(*args, **kwargs):
    pytest.skip(reason=(
        "Likely upstream issue with hdf5storage. "
        "For details, see https://github.com/stes/neural_cl/issues/417"))(
            *args, **kwargs)


def register(*file_endings):
    def _register(f):
    return _register

def generate_numpy(filename):
    A = np.arange(1000).reshape(10, 100)

def generate_numpy_confounder(filename):
    A = np.arange(1000).reshape(10, 100)

def generate_h5(filename):








    _skip_hdf5storage()
    _skip_hdf5storage()
    _skip_hdf5storage()
    _skip_hdf5storage()
    _skip_hdf5storage()
    _skip_hdf5storage()
    _skip_hdf5storage()
def test_load(save_data):

    with pytest.raises((AttributeError, TypeError)):

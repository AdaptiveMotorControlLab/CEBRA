import argparse
import glob
import itertools

_sweeps = {}
def add_sweep(fn):
    """Decorate a function to add it as a sweep config."""
    global _sweeps
    _sweeps[fn.__name__] = fn

@add_sweep
def dino():
    """Compute features using the DINO model."""
    def log_exists(video, model, patchsize):
        fnames = glob.glob(f"features/allen_movies/{model}/{patchsize}/{video}/testfeat.pth")
        return len(fnames) > 0
    
    def allen_movies():
        videos = ["movie_one_image_stack.npz", "movie_two_image_stack.npz", "movie_three_image_stack.npz"]
    
        # models = ["deit_tiny", "vit_base", "deit_small" ]
        # patchsizes = [8, 16]
    
        models = [ "vit_small", "vit_base" ]
        patchsizes = [ 8, 16 ]
        for params in itertools.product(videos, models, patchsizes):
            if not log_exists(*params):
                yield params

    for params in sorted(allen_movies()):
        print(" ".join(map(str, params)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sweep', choices = list(_sweeps.keys()))
    args = parser.parse_args()
    fn = _sweeps.get(args.sweep)
    fn()

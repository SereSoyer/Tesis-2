"""Build split-specific data readers from a shared preprocessing configuration."""
from pathlib import Path
from src.datasets.classification import ClassificationDataset, make_loader
from src.preprocessing.multispectral import Preprocess

class CheckedPreprocess:
    def __init__(self, config):
        self.shape = tuple(config['expected_source_shape'])
        self.transform = Preprocess(config['channels'], config['nan_policy'], config['normalization'])

    def __call__(self, array):
        if array.shape != self.shape:
            raise ValueError(f'Expected source HWC shape {self.shape}, got {array.shape}')
        return self.transform(array)

def configured_dataset(config, root, split):
    if split not in ('train', 'val', 'test'):
        raise ValueError('split must be train, val or test')
    if config['normalization'] != 'none':
        raise ValueError('This L4S configuration preserves the distributed scaling; additional normalization requires explicit implementation')
    ds = ClassificationDataset(Path(root)/config['manifests'][split], root,
                               CheckedPreprocess(config), config['image_key'], config['layout'])
    expected_split = {'train':'TrainData', 'val':'ValidData', 'test':'TestData'}[split]
    if any(r['dataset'] != config['dataset'] or r.get('source_split') != expected_split for r in ds.rows):
        raise ValueError('Manifest dataset/source_split does not match requested split')
    return ds

def configured_loader(config, root, split):
    return make_loader(configured_dataset(config, root, split),
                       batch_size=config['batch_size'], training=split=='train',
                       seed=config['seed'], num_workers=config['num_workers'])

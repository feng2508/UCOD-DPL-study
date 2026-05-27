import os
from pathlib import Path

_DEFAULT_CACHE_DIR = (
    '/kaggle/temp/UCOD-DPL/cache'
    if Path('/kaggle/working').exists()
    else './datasets/cache'
)
_CACHE_DIR = os.environ.get('UCOD_CACHE_DIR', _DEFAULT_CACHE_DIR)

cfg = dict(
    dataset_cfg = dict(
        cache_dir=os.environ.get('UCOD_LOOK_TWICE_CACHE_DIR', f'{_CACHE_DIR}/look_twice'),
        dataset_dir='./datasets/RefCOD',
        trainset_cfg = dict(
            DATASET='TR-CAMO+TR-COD10K',
            require_label=False,
        ),
    valset_cfg = dict(
            DATASET='TE-COD10K',
            require_label=True,
        ),
    )     
)

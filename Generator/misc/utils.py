import os
import torch
import torchaudio
from pathlib import Path


class AudioFolder(torch.utils.data.Dataset):
    def __init__(self, path):
        self._path = path
        self._parse_filesystem()

    def _parse_filesystem(self):
        if not os.path.isdir(self._path):
            raise RuntimeError(
                'Dataset not found.'
            )

        self._walker = sorted(str(p.stem) for p in Path(self._path).glob('*.wav'))

    def _load_item(self, fileid: str, path: str):
        file_audio = os.path.join(path, fileid + '.wav')
        waveform, sample_rate = torchaudio.load(file_audio)
        return waveform, sample_rate

    def __getitem__(self, n: int):
        fileid = self._walker[n]
        item = self._load_item(fileid, self._path)
        return item

    def __len__(self):
        return len(self._walker)


def build_folder_structure(iter_num):
    try:
        os.mkdir('runs/')
    except FileExistsError:
        pass

    try:
        os.mkdir('runs/iter_{}/'.format(iter_num))
    except FileExistsError:
        pass

    try:
        os.mkdir('runs/iter_{}/output/'.format(iter_num))
    except FileExistsError:
        pass

    try:
        os.mkdir('runs/iter_{}/output/'.format(iter_num))
    except FileExistsError:
        pass

    try:
        os.mkdir('runs/iter_{}/model/'.format(iter_num))
    except FileExistsError:
        pass


def get_iter():
    if os.path.isdir('runs/'):
        iters = os.listdir('runs/')
        iter_num = len(iters) + 1
        return iter_num
    else:
        # Returns iter 1 if runs directory doesn't exist
        return 1


def profiled_function(fn):
    def decorator(*args, **kwargs):
        with torch.autograd.profiler.record_function(fn.__name__):
            return fn(*args, **kwargs)
    decorator.__name__ = fn.__name__
    return decorator


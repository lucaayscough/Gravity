import os
import torch
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path


class AudioFolder(Dataset):
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


def gradient_penalty(discriminator, real, fake, device = 'cpu'):
    batch_size, channels, samples = real.shape
    epsilon = torch.rand((batch_size, 1,  1)).repeat(1, channels, samples).to(device)
    epsilon = (epsilon - 0.5) * 2

    real.requires_grad = True
    
    interpolated_sounds = real * epsilon + fake * (1 - epsilon)
    mixed_scores = discriminator(interpolated_sounds)

    gradient = torch.autograd.grad(
        inputs = interpolated_sounds,
        outputs = mixed_scores,
        grad_outputs = torch.ones_like(mixed_scores),
        create_graph = True,
        retain_graph = True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim = 1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty


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

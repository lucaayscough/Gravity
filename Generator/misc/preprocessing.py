import os
import torch
import torchaudio
from torchaudio import transforms
from torch.utils.data import DataLoader
from utils import AudioFolder
from pydub import AudioSegment, effects  


class Preprocessing:
    def __init__(self, pre_path, post_path):
        self._pre_path = pre_path
        self._post_path = post_path
        
        dataset = AudioFolder(self._pre_path)
        self._dataloader = DataLoader(dataset, batch_size = 1, shuffle = False)


    def process(self, sample_rate, fade_out_len, max_len):
        counter = 0
        for data in self._dataloader:
            print(data[0].shape, counter)

            if data[1] < sample_rate:
                continue
            if data[1] > sample_rate:
                new_file = torchaudio.transforms.Resample(orig_freq = data[1], new_freq = sample_rate)(data[0])
            else:
                new_file = data[0]
            
            new_file = new_file[0][0]    # If stereo -> mono
            if len(new_file) > max_len:
                new_file = new_file[:max_len]    # Clip file length
                new_file = transforms.Fade(fade_out_len = fade_out_len)(new_file)  # Add fade to file end
            elif len(new_file) < max_len:
                empty_tensor = torch.zeros(max_len - len(new_file))
                new_file = torch.cat((new_file, empty_tensor))
            new_file = new_file.view((1, -1))    # Add channel num dimension
            torchaudio.save(filepath = self._post_path + str(counter) + '.wav', src = new_file, sample_rate = sample_rate) # Save file
            
            # Normalize audio
            rawsound = AudioSegment.from_file(self._post_path + str(counter) + '.wav', "wav")  
            normalizedsound = effects.normalize(rawsound, 2.0)  
            normalizedsound.export(self._post_path + str(counter) + '.wav', format="wav")

            counter += 1
    

if __name__ == '__main__':
    pp = Preprocessing('H:/dev/Gravity/Generator/datasets/dataset_2/', 'H:/dev/Gravity/Generator/datasets/dataset_3/')
    pp.process(sample_rate = 44100, fade_out_len = 8000, max_len = 16384)

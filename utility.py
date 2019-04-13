import numpy as np
import pyworld as pw
# import soundfile as sf
import tensorflow as tf
import os,shutil
import glob

father_folder = os.path.dirname(os.path.abspath(__file__))

def get_speakers():

    '''return current selected singers for training
    '''
    trainset = os.path.join(father_folder, 'data/singers')

    all_speaker= os.listdir(trainset)
    all_speaker.sort()
    print(f'get speakers:{all_speaker}')
    return all_speaker


class Normalizer(object):
    '''Normalizer: convience method for fetch normalize instance'''
    def __init__(self):
        self.all_speaker = get_speakers()
        statfolderpath = os.path.join(father_folder, 'etc')
        self.folderpath = statfolderpath
        self.norm_dict = self.normalizer_dict()

    def forward_process(self, x, speakername):
        mean = self.norm_dict[speakername]['coded_sps_mean']
        std = self.norm_dict[speakername]['coded_sps_std']
        mean = np.reshape(mean, [-1,1])
        std = np.reshape(std, [-1,1])
        x = (x - mean) / std

        return x

    def backward_process(self, x, speakername):
        mean = self.norm_dict[speakername]['coded_sps_mean']
        std = self.norm_dict[speakername]['coded_sps_std']
        mean = np.reshape(mean, [-1,1])
        std = np.reshape(std, [-1,1])
        x = x * std + mean

        return x

    def normalizer_dict(self):
        '''return all speakers normailzer parameter'''
        d = {}
        print(self.folderpath)
        for one_speaker in self.all_speaker:

            p = os.path.join(self.folderpath, '*.npz')
            try:
                stat_filepath = [fn for fn in glob.glob(p) if one_speaker in fn][0]
            except:
                raise Exception('====no match files!====')
            print(f'found stat file: {stat_filepath}')
            t = np.load(stat_filepath)
            d_temp = t.f.arr_0.item()            
            d[one_speaker] = d_temp

        return d
    
    def pitch_conversion(self, f0, source_speaker, target_speaker):
        '''Logarithm Gaussian normalization for Pitch Conversions'''
        mean_log_src = self.norm_dict[source_speaker]['log_f0s_mean']
        std_log_src = self.norm_dict[source_speaker]['log_f0s_std']

        mean_log_target = self.norm_dict[target_speaker]['log_f0s_mean']
        std_log_target = self.norm_dict[target_speaker]['log_f0s_std']

        f0_converted = np.exp((np.ma.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

        return f0_converted
    
class GenerateStatics(object):
    def __init__(self):
        self.folder = os.path.join(father_folder, 'data/processed')
        self.all_speaker = get_speakers()

        self.include_dict = {}
        for s in self.all_speaker:
            if not self.include_dict.__contains__(s):
                self.include_dict[s] = []

            for one_file in os.listdir(self.folder):
                if one_file.startswith(s) and one_file.endswith('npy'):
                    self.include_dict[s].append(one_file)

        self.include_dict_npz = {}
        for s in self.all_speaker:
            if not self.include_dict_npz.__contains__(s):
                self.include_dict_npz[s] = []

            for one_file in os.listdir(self.folder):
                if one_file.startswith(s) and one_file.endswith('npz'):
                    self.include_dict_npz[s].append(one_file)

    @staticmethod
    def coded_sp_statistics(coded_sps):
        # sp shape (T, D)
        coded_sps_concatenated = np.concatenate(coded_sps, axis = 1)
        coded_sps_mean = np.mean(coded_sps_concatenated, axis = 1, keepdims = False)
        coded_sps_std = np.std(coded_sps_concatenated, axis = 1, keepdims = False)
        return coded_sps_mean, coded_sps_std

    @staticmethod
    def logf0_statistics(f0s):
        log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
        log_f0s_mean = log_f0s_concatenated.mean()
        log_f0s_std = log_f0s_concatenated.std()

        return log_f0s_mean, log_f0s_std

    def generate_stats(self, statfolder: str = './etc'):
        '''generate all user's statitics used for calutate normalized
           input like sp, f0
           step 1: generate coded_sp mean std
           step 2: generate f0 mean std
         '''
        etc_path = os.path.join(father_folder, statfolder)
        if not os.path.exists(etc_path):
                os.makedirs(etc_path, exist_ok=True)
        
        for one_speaker in self.include_dict.keys():
            coded_sps = []
            
            arr = self.include_dict[one_speaker]
            if len(arr) == 0:
                continue
            for one_file in arr:
                t = np.load(os.path.join(self.folder, one_file))
                coded_sps.append(t)

            coded_sps_mean, coded_sps_std = self.coded_sp_statistics(coded_sps)


            f0s = []            
            arr01 = self.include_dict_npz[one_speaker]
            if len(arr01) == 0:
                continue
            for one_file in arr01:
                t = np.load(os.path.join(self.folder, one_file))
                d =  t.f.arr_0.item()
                f0_ = np.reshape(d['f0'], [-1,1])
                f0s.append(f0_)
            log_f0s_mean, log_f0s_std = self.logf0_statistics(f0s)
            print(log_f0s_mean, log_f0s_std)

            tempdict = {
                'log_f0s_mean':log_f0s_mean,
                'log_f0s_std':log_f0s_std,
                'coded_sps_mean':coded_sps_mean,
                'coded_sps_std':coded_sps_std
            }
        
            filename = os.path.join(etc_path, f'{one_speaker}-stats.npz')
            print(f'save: {filename}')
            np.savez(filename, tempdict)

if __name__ == "__main__":
    print(get_speakers())
import io,os
import shutil
import time, sys
import shlex, subprocess
import glob

father_folder = os.path.dirname(os.path.abspath(__file__))

#F蔡琴 F高胜美  M刀郎 M刘德华
source = ['F蔡琴', 'M刀郎', 'F高胜美', 'M刘德华']
target = ['M刀郎', 'F蔡琴', 'F蔡琴', 'M刀郎']


def get_model_dir():
    #automatically find model's directory
        t_dir = os.path.join(father_folder, 'out')
        t_folder = [x for x in os.listdir(t_dir) if x.startswith('100')][0]
        model_dir = os.path.join(t_dir, t_folder,'model')

        return model_dir


model_dir = get_model_dir()
print(model_dir)

def one_to_one_convert():
    for s, t in zip(source, target):
            print(s,t)
            cmdstr = f'python convert.py --source_speaker {s} --target_speaker {t} --model_dir {model_dir}'

            analyzer_args = shlex.split(cmdstr)
            q = subprocess.run(analyzer_args)

            assert q.returncode == 0

def many_to_many_convert():
    for t in source:
            for s in source:
                cmdstr = f'python convert.py --source_speaker {s} --target_speaker {t} --model_dir {model_dir}'

                analyzer_args = shlex.split(cmdstr)
                q = subprocess.run(analyzer_args)

                assert q.returncode == 0


if __name__ == '__main__':
    many_to_many_convert()

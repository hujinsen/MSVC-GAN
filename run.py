import io,os
import shutil
import time, sys
import shlex, subprocess
import glob

#运行前确认，已经处理过得就不要在运行
cmd01 = 'python preprocess.py'
args01 = shlex.split(cmd01)
subprocess.run(args01)

# #需要检查model.py是否运行的是正确的模型
cmd02 = 'python train.py'
args02 = shlex.split(cmd02)
subprocess.run(args02)

# #检查转换方式是否正确，模型是否正确
cmd03 = 'python convert_all.py'
args03 = shlex.split(cmd03)
subprocess.run(args03)

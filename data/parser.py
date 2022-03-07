from email.mime import base
import torch 
import torchaudio

import numpy as np 
import matplotlib.pyplot as plt 
import os 
import time 
import pandas as pd 
from tqdm import tqdm 

def parse_waveform(file_loc, vggish_net = None, **kwargs):
  if vggish_net==None: vggish_net = torch.hub.load('harritaylor/torchvggish', 'vggish')
  vggish_net.eval()

  try:
    x = vggish_net(file_loc).cpu().detach().numpy()
    x = np.squeeze(x)

    return x 

  except RuntimeError as e:
    print(file_loc)
    w, _ = torchaudio.load(file_loc)
    print(w.shape)
    print(e)
    
    return None 

# TODO: redo so dataset is already train/test split
def parse_waveforms(base_dir, split, **kwargs):
  meta = pd.read_csv(f'{base_dir}/metadata.csv')
  datas, labels = [], []
  durations = [] 
  vggish_net = torch.hub.load('harritaylor/torchvggish', 'vggish')
  vggish_net.eval()
  nums = 0
  total_nums = 0

  for f in tqdm(os.listdir(base_dir+"/"+split), total=len(os.listdir(base_dir+"/"+split))):
    if f.split(".")[-1]!='wav': continue
    total_nums +=1
    try:
      x = vggish_net(f"{base_dir}/{split}/{f}").cpu().detach().numpy()
      x = np.squeeze(x)

      if len(x.shape)>1:
        for i in range(x.shape[0]):
          datas.append(np.squeeze(x[i]))
          labels.append(int(meta[meta['slice_file_name']==f]['classID']))
      else:
        datas.append(np.squeeze(x))
        labels.append(int(meta[meta['slice_file_name']==f]['classID']))

      nums += 1

      w, s = torchaudio.load(f"{base_dir}/{split}/{f}")
      durations.append(w.shape[-1]/s)

    except RuntimeError as e:
      print(f)
      w, _ = torchaudio.load(f"{base_dir}/{split}/{f}")
      print(w.shape)
      print(e)

    #if nums>=100: break 
  
  print(f"{nums}/{total_nums} were successfully inferenced")

  datas = [torch.tensor(a) for a in datas]
  labels = [int(a) for a in labels]

  return datas, labels
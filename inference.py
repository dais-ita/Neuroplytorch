import torch
from torch.nn import functional as F 
import torchvision
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import ModelCheckpoint

import os 
import argparse
import yaml 
import pickle 
from tqdm import tqdm 
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 

from data.parser import parse_waveforms, parse_waveform

#.datasets import MNISTDataModule, EndToEndDataModule, EndToEndNoTestDataModule, ReasoningDataModule, fetch_perception_data
from training import models, basic_models#BasicLSTM, MNISTModel, Neuroplytorch, ReasoningModel, MNISTWindow
from data import data, datasets

def get_complex_parameters(complex_events_dict) -> tuple: 
    ce_fsm_list, ce_time_list = [], [] 
    for k in complex_events_dict.keys():
        complex_event = complex_events_dict[k]
        ce_fsm_list.append(torch.tensor(complex_event['PATTERN'])) 
        max_time = [float('inf') if a=='INF' else a for a in complex_event['MAX_TIME']]
        ce_time_list.append(torch.tensor([max_time, complex_event['EVENTS_BETWEEN']]))
    
    return ce_fsm_list, ce_time_list

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
 
if __name__=="__main__":
    parser = argparse.ArgumentParser() 

    # This distinguishes between problems, i.e. the different scenarios, pattern parameters etc.
    parser.add_argument('--name', dest='config_name', type=str, default='basic_neuro_experiment')
    args = vars(parser.parse_args())

    # TODO: on run, save the config file as hyperparameters for the logger
    with open(f'./configs/{args["config_name"]}.yaml') as file:
        x = yaml.load(file, Loader=yaml.FullLoader)
        training = x['TRAINING'] 
        complex_events = x['COMPLEX EVENTS']

        ce_fsm_list, ce_time_list = get_complex_parameters(complex_events)
        assert(data.check_complex_parameters(ce_fsm_list, ce_time_list), "Pattern and temporal metadata don't match, check the config file")

        MODULE_NAME = args['config_name']

        perception_model_args = training['PERCEPTION']['PARAMETERS'].get('MODEL', {})
        reasoning_model_args = training['REASONING']['PARAMETERS'].get('MODEL', {})
        end_to_end_model_args = training['NEUROPLYTORCH']['PARAMETERS'].get('MODEL', {})

        perception_dataset_args = training['PERCEPTION']['PARAMETERS'].get('DATASET', {})
        reasoning_dataset_args = training['REASONING']['PARAMETERS'].get('DATASET', {})
        end_to_end_dataset_args = training['NEUROPLYTORCH']['PARAMETERS'].get('DATASET', {})

        perception_loss_str = training['PERCEPTION']['PRETRAIN'].get('LOSS_FUNCTION', 'MSELoss')
        reasoning_loss_str = training['REASONING'].get('LOSS_FUNCTION', 'MSELoss')

        pretrain_perception = training['PERCEPTION']['PRETRAIN'].get('PRETRAIN_PERCEPTION', False)
        pretrain_num_epochs = training['PERCEPTION']['PRETRAIN'].get('PRETRAIN_EPOCHS', 10)

        pretrain_lr = training['PERCEPTION']['PRETRAIN'].get('LEARNING_RATE', 0.001)
        reasoning_lr = training['REASONING'].get('LEARNING_RATE', 0.001)

        reasoning_epochs = training['REASONING']['EPOCHS']
        reasoning_num_data = training['REASONING']['EPOCHS']

        end_to_end_lr = training['NEUROPLYTORCH'].get('LEARNING_RATE', 0.001)
        end_to_end_loss_str = training['NEUROPLYTORCH'].get('LOSS_FUNCTION', 'MSELoss')
        end_to_end_epochs = training['NEUROPLYTORCH']['EPOCHS']
        
        no_test = end_to_end_dataset_args.get('no_test', True)

        window_size = training.get('WINDOW_SIZE', 10)
        num_primitive_events = training.get('NUM_PRIMITIVE_EVENTS', 10)
        input_size = perception_model_args['input_size']
        perception_model_args.pop('input_size')


        new_reasoning = basic_models.BasicLSTM(input_size=num_primitive_events, output_size=len(ce_fsm_list), loss_str=reasoning_loss_str, lr=reasoning_lr)
        new_perception = basic_models.get_model(training['PERCEPTION']['MODEL'])(input_size=input_size, output_size=num_primitive_events, **perception_model_args)
        new_reasoning.load_state_dict(torch.load(f'models/neuroplytorch/{reasoning_loss_str}_{MODULE_NAME}/reasoning_model.pt')) 
        new_perception.load_state_dict(torch.load(f'models/neuroplytorch/{reasoning_loss_str}_{MODULE_NAME}/perception_model.pt')) 

        new_perception = models.PerceptionWindow(new_perception, window_size=window_size, num_primitive_events=num_primitive_events)
        end_model = models.Neuroplytorch(reasoning_model=new_reasoning, perception_model=new_perception, window_size=window_size, num_primitive_events=num_primitive_events,
            loss_str=end_to_end_loss_str, lr=end_to_end_lr)
        
        #x = [] 
        #if training['DATASET']['TYPE']=='Pytorch Dataset':
        #    x = datasets.fetch_perception_data(dataset_str=training['DATASET']['NAME'], dataset_loc=training['DATASET']['LOCATION'])
        #else:
        #    x = datasets.fetch_perception_data_local(dataset_loc=training['DATASET']['LOCATION'], dataset_type=training['DATASET']['TYPE'], **perception_dataset_args)


        #end_data = datasets.EndToEndNoTestDataModule (dataset=x, ce_fsm_list=ce_fsm_list, ce_time_list=ce_time_list, num_primitive_events=num_primitive_events, 
        #        window_size=window_size, **end_to_end_dataset_args)

        #end_data.prepare_data()
        #end_data.setup(stage='fit')

        #trainer = pl.Trainer(max_epochs=end_to_end_epochs, gpus=1, precision=16)
        #trainer.test(end_model, dataloaders=end_data.val_dataloader())

        

        #xs, ys = [], [] 
        #for xx in x['test']:
        #    d, l = xx
        #    xs.append(d)
        #    ys.append(l)


        #ws, ls, cs, ss = [], [], [], [] 

        #for i in tqdm(range(0, len(xs)-window_size), total=len(xs)-window_size):
        #    curr_window, curr_label = torch.stack(xs[i:i+window_size]), torch.stack([data.create_primitive_event(num_primitive_events,a) for a in ys[i:i+window_size]])
        #    curr_complex = data.get_complex_label(curr_label, ce_fsm_list, ce_time_list, count_windows=False)
        #    curr_simple = data.complex_to_simple(curr_complex)

        #    ws.append(curr_window)
        #    ls.append(curr_label)
        #    cs.append(curr_complex)

        #confs = [] 
            
        #for w, l, c in zip(ws, ls, cs):
        #    o_i, o_r = end_model(w[None, :])
        #    confs.append(torch.max(o_r).item())

        #plt.plot(confs)
        #plt.show() 

        class_id_to_name = {} 

        meta = pd.read_csv('datasets/UrbanSound8K/metadata.csv')
        for i, v in meta.iterrows():
            class_id_to_name[v['classID']] = class_id_to_name.get(v['classID'], v['class'])

        x = parse_waveform('datasets/UrbanSound8K/demo_audio_base.wav')
        #x = parse_waveform('datasets/UrbanSound8K/demo_audio_gunshots.wav')
        xs = torch.tensor(x)

        confs = [] 
        times = [8.1] 
        for i in tqdm(range(0, len(xs)-window_size), total=len(xs)-window_size):
            curr_window = xs[i:i+window_size]
            times.append(times[-1]+0.9)
            o_i, o_r = end_model(curr_window[None, :])

            confs.append(o_r.cpu().detach().numpy())

        print(len(confs))
        exit() 
        times = times[1:]
        confs = np.array(confs)
        confs = np.squeeze(confs)
        confs = np.swapaxes(confs, 0, 1)
        legs = [] 
        
        for i in range(confs.shape[0]):
            class_name = class_id_to_name[i]
            if class_name not in ['street_music', 'siren', 'children_playing', 'gun_shot']: continue
            plt.plot(times, moving_average(confs[i,:], 1))
            legs.append(class_id_to_name[i])
            
        
        print(class_id_to_name)
        plt.legend(legs)
        plt.show()
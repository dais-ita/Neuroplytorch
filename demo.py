from ast import parse
import torch
from torch.nn import functional as F 
import torchvision
import torchaudio
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
import pyaudio
import wave 
import sounddevice as sd 
import time 

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
    parser.add_argument('--file', dest='file_loc', type=str, default='datasets/UrbanSound8K/demo_audio_base.wav')
    parser.add_argument('--mic', dest='use_mic', type=int, default=0)
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

        perception_loss_str = training['PERCEPTION'].get('PRETRAIN', {}).get('LOSS_FUNCTION', 'MSELoss')
        reasoning_loss_str = training['REASONING'].get('LOSS_FUNCTION', 'MSELoss')

        pretrain_perception = training['PERCEPTION'].get('PRETRAIN', {}).get('PRETRAIN_PERCEPTION', False)
        pretrain_num_epochs = training['PERCEPTION'].get('PRETRAIN', {}).get('PRETRAIN_EPOCHS', 10)

        pretrain_lr = training['PERCEPTION'].get('PRETRAIN', {}).get('LEARNING_RATE', 0.001)
        reasoning_lr = training['REASONING'].get('LEARNING_RATE', 0.001)

        reasoning_epochs = training['REASONING']['EPOCHS']
        reasoning_num_data = training['REASONING']['EPOCHS']

        end_to_end_lr = training['NEUROPLYTORCH'].get('LEARNING_RATE', 0.001)
        end_to_end_loss_str = training['NEUROPLYTORCH'].get('LOSS_FUNCTION', 'MSELoss')
        end_to_end_epochs = training['NEUROPLYTORCH']['EPOCHS']
        
        no_test = end_to_end_dataset_args.get('no_test', True)

        window_size = training.get('WINDOW_SIZE', 10)
        num_primitive_events = training.get('NUM_PRIMITIVE_EVENTS', 10)
        input_size = training['PERCEPTION']['INPUT_SIZE']


        new_reasoning = basic_models.BasicLSTM(input_size=num_primitive_events, output_size=len(ce_fsm_list), loss_str=reasoning_loss_str, **reasoning_model_args)
        new_perception = basic_models.get_model(training['PERCEPTION']['MODEL'])(input_size=input_size, output_size=num_primitive_events, **perception_model_args)
        new_reasoning.load_state_dict(torch.load(f'models/neuroplytorch/{reasoning_loss_str}_{MODULE_NAME}/reasoning_model.pt')) 
        new_perception.load_state_dict(torch.load(f'models/neuroplytorch/{reasoning_loss_str}_{MODULE_NAME}/perception_model.pt')) 

        new_perception = models.PerceptionWindow(new_perception, window_size=window_size, num_primitive_events=num_primitive_events)
        end_model = models.Neuroplytorch(reasoning_model=new_reasoning, perception_model=new_perception, window_size=window_size, num_primitive_events=num_primitive_events,
            loss_str=end_to_end_loss_str, lr=end_to_end_lr)




        class_id_to_name = {} 

        meta = pd.read_csv('datasets/UrbanSound8K/metadata.csv')
        for i, v in meta.iterrows():
            class_id_to_name[v['classID']] = class_id_to_name.get(v['classID'], v['class'])

        if args['use_mic']==1:
            #AUDIO INPUT
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 44100
            CHUNK = 1024
            RECORD_SECONDS = 1
            WAVE_OUTPUT_FILENAME = "output.wav"

            audio = pyaudio.PyAudio()
            vggish_net = torch.hub.load('harritaylor/torchvggish', 'vggish')

            # start Recording
            stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)

            past_xs = [] 
            print("recording")
            while(1):
                frames = []
                for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                    data = stream.read(CHUNK)
                    frames.append(data)
                waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                waveFile.setnchannels(CHANNELS)
                waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                waveFile.setframerate(RATE)
                waveFile.writeframes(b''.join(frames))
                waveFile.close()
                spf = wave.open(WAVE_OUTPUT_FILENAME,'r')

                #Extract Raw Audio from Wav File
                signal = spf.readframes(-1)
                signal = np.fromstring(signal, dtype=np.int16)
                copy= signal.copy()

                x = parse_waveform('output.wav', vggish_net=vggish_net)
                xs = torch.tensor(x)
                past_xs.append(xs)
                past_xs = past_xs[-10:-1:1] + [past_xs[-1]]
                
                if len(past_xs)==10:
                    xs = torch.stack(past_xs)
                    o_i, o_r = end_model(xs[None, :])
                    for i in range(len(class_id_to_name.keys())):
                        print(f"{class_id_to_name[i]}:", end='')
                        for i in range(int(o_r[0][i].item()/0.1)): print("=", end='')
                        print()

                    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

        elif args['use_mic']==2:
            #AUDIO INPUT
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 44100
            CHUNK = 1024
            RECORD_SECONDS = 1
            WAVE_OUTPUT_FILENAME = "output.wav"
            INPUT_FILENAME = args['file_loc']
            INPUT_WAV, RATE = torchaudio.load(INPUT_FILENAME)

            audio = pyaudio.PyAudio()
            vggish_net = torch.hub.load('harritaylor/torchvggish', 'vggish')

            past_xs = [] 
            print("recording")
            time.sleep(10)
            for i in range(120):
                frames = INPUT_WAV[:,i*RATE:(i+1)*RATE]
                torchaudio.save('output.wav', frames, RATE, encoding="PCM_S", bits_per_sample=16)
                spf = wave.open(WAVE_OUTPUT_FILENAME,'r')

                #Extract Raw Audio from Wav File
                signal = spf.readframes(-1)
                signal = np.fromstring(signal, dtype=np.int16)
                copy= signal.copy()

                x = parse_waveform('output.wav', vggish_net=vggish_net)
                xs = torch.tensor(x)
                past_xs.append(xs)
                past_xs = past_xs[-10:-1:1] + [past_xs[-1]]
                
                if len(past_xs)==10:
                    xs = torch.stack(past_xs)
                    o_i, o_r = end_model(xs[None, :])
                    for i in range(len(class_id_to_name.keys())):
                        print(f"{class_id_to_name[i]}:", end='')
                        for i in range(int(o_r[0][i].item()/0.1)): print("=", end='')
                        print()

                    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

                
                sd.play(torch.mean(frames, axis=0).detach().numpy(), RATE)
                time.sleep(0.9)

        else:
            #x = parse_waveform('datasets/UrbanSound8K/demo_audio_base.wav')
            x = parse_waveform(args['file_loc'])
            xw, sr = torchaudio.load(args['file_loc'])
            xs = torch.tensor(x)

            
            duration = xw.size()[1]/sr
            duration_per_clip = duration/xs.size()[0]

            confs = [] 
            for i in tqdm(range(0, len(xs)-window_size+1), total=len(xs)-window_size+1):
                curr_window = xs[i:i+window_size]
                
                if curr_window.size()[0]<10: break
                o_i, o_r = end_model(curr_window[None, :])

                confs.append(o_r.cpu().detach().numpy())

            confs = np.array(confs)
            confs = np.squeeze(confs)
            confs = np.swapaxes(confs, 0, 1)
            legs = [] 
            ma = 1
            times = range((10)+(ma-1), xs.size()[0]+1)
            actual_times = [duration_per_clip * i for i in times]
            print(actual_times)
            print(duration_per_clip)

            demo_outs = {}
            
            for i in range(confs.shape[0]):
                class_name = class_id_to_name[i]
                if class_name not in ['street_music', 'siren', 'children_playing', 'gun_shot']: continue
                ma_confs = moving_average(confs[i,:], ma)
                plt.plot(times, ma_confs)
                legs.append(class_id_to_name[i])

                demo_outs[i] = {
                    'class': class_name,
                    'times': actual_times,
                    'confs': ma_confs
                }
                
            
            pickle.dump(demo_outs, open('base.p', 'wb'))
            print(class_id_to_name)
            plt.legend(legs)
            plt.ylabel('Confidence')
            plt.xlabel('Time /s')
            plt.show()
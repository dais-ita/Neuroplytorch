import torch
import torchvision
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import ModelCheckpoint

import os 
import argparse
import yaml 
import pickle 

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
 
if __name__=="__main__":
    parser = argparse.ArgumentParser() 

    # This distinguishes between problems, i.e. the different scenarios, pattern parameters etc.
    parser.add_argument('--name', dest='config_name', type=str, default='basic_neuro_experiment')
    parser.add_argument('--logic', dest='check_logic', type=int, default=0) # if False, run end-to-end, if True run logic_check on reasoning layer

    args = vars(parser.parse_args())

    # TODO: on run, save the config file as hyperparameters for the logger
    with open(f'./configs/{args["config_name"]}.yaml') as file:
        x = yaml.load(file, Loader=yaml.FullLoader)
        training = x['TRAINING'] 
        complex_events = x['COMPLEX EVENTS']

        ce_fsm_list, ce_time_list = get_complex_parameters(complex_events)
        assert(data.check_complex_parameters(ce_fsm_list, ce_time_list), "Pattern and temporal metadata don't match, check the config file")

        MODULE_NAME = args['config_name']

        # TODO: redo and double check these, make sure **kwargs are going in the right places etc. 
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
        input_size = perception_model_args.pop('input_size', None)
       

        # fetch raw input data 
        x = [] 
        if training['DATASET']['TYPE']=='Pytorch Dataset':
            x = datasets.fetch_perception_data(dataset_str=training['DATASET']['NAME'], dataset_loc=training['DATASET']['LOCATION'])
        else:
            x = datasets.fetch_perception_data_local(dataset_loc=training['DATASET']['LOCATION'], dataset_type=training['DATASET']['TYPE'], **perception_dataset_args)




        # if pretrain_perception then train the perception model before attaching to Neuroplytorch, else leave untrained
        perception_model = basic_models.get_model(training['PERCEPTION']['MODEL'])(input_size=input_size, output_size=num_primitive_events, **perception_model_args)
        if input_size==None: perception_model = basic_models.get_model(training['PERCEPTION']['MODEL'])(output_size=num_primitive_events, **perception_model_args)
        if pretrain_perception:
            perception_data = datasets.get_datamodule(training['PERCEPTION']['PRETRAIN']['DATA_MODULE'])(data_dir=training['DATASET']['NAME'], **perception_dataset_args)
            model = models.get_model(training['PERCEPTION']['PRETRAIN']['MODEL_MODULE'])(loss_str=perception_loss_str, lr=pretrain_lr, **perception_model_args)
            trainer = pl.Trainer(max_epochs=pretrain_num_epochs, gpus=1, precision=16) 
            trainer.fit(model, perception_data)

            perception_model = model.model

        perception_model = models.PerceptionWindow(perception_model=perception_model, window_size=window_size, num_primitive_events=num_primitive_events)




        # if a reasoning model already exists
        if os.path.exists(f'./models/reasoning/reasoning_model_{reasoning_loss_str}_{MODULE_NAME}.pt'):
            reasoning_model = models.ReasoningModel(input_size=num_primitive_events, output_size=len(ce_fsm_list), loss_str=reasoning_loss_str, lr=reasoning_lr)
            reasoning_model = reasoning_model.model
            reasoning_model.load_state_dict(torch.load(f'./models/reasoning/reasoning_model_{reasoning_loss_str}_{MODULE_NAME}.pt'))

        # otherwise synthesise data and train reasoning model separate from Neuroplytorch model
        else:
            reasoning_data = datasets.ReasoningDataModule(ce_fsm_list=ce_fsm_list, ce_time_list=ce_time_list, num_primitive_events=num_primitive_events, 
                window_size=window_size, **reasoning_dataset_args)

            model = models.ReasoningModel(input_size=num_primitive_events, output_size=len(ce_fsm_list), loss_str=reasoning_loss_str, lr=reasoning_lr)
            trainer = pl.Trainer(max_epochs=reasoning_epochs, gpus=1, precision=16)

            trainer.fit(model, reasoning_data)
            trainer.test(model, reasoning_data)

            model.save_weights(f'./models/reasoning/reasoning_model_{reasoning_loss_str}_{MODULE_NAME}.pt')
            reasoning_model = model.model 
            try: os.remove('curr_tmp_reasoning_model.pt')
            except Exception: pass 




        if args['check_logic']:
            models.check_reasoning_logic(reasoning_model, ce_fsm_list, ce_time_list, num_primitive_events, window_size)

        else:

            # Push raw data with pattern parameters into an end-to-end dataset (NoTest implies the test set is used as validation)
            no_test_args = end_to_end_dataset_args.pop('no_test', True)
            end_data = datasets.EndToEndNoTestDataModule if no_test_args else datasets.EndToEndDataModule
            end_data = end_data(dataset=x, ce_fsm_list=ce_fsm_list, ce_time_list=ce_time_list, num_primitive_events=num_primitive_events, 
                window_size=window_size, **end_to_end_dataset_args)

            # create a Neuroplytorch model from the reasoning model and perception model from previous and train
            end_model = models.Neuroplytorch(reasoning_model=reasoning_model, window_size=window_size, num_primitive_events=num_primitive_events,loss_str=end_to_end_loss_str, 
                perception_model=perception_model, lr=end_to_end_lr)

            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
                dirpath="checkpoints",
                filename=f"{MODULE_NAME}"+"-{epoch:02d}-{val_loss:.2f}",
                save_top_k=3,
                mode="min",
            )
            trainer = pl.Trainer(max_epochs=end_to_end_epochs, gpus=1, precision=16)
            trainer.fit(end_model, end_data)

            end_model.save_model(f'models/neuroplytorch/{reasoning_loss_str}_{MODULE_NAME}')

            
            

            # TODO: Document kwargs
            # TODO: SOCIAL MEDIA SCENARIO: https://arxiv.org/pdf/1709.01848.pdf


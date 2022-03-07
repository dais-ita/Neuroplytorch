from logging import log
from typing import OrderedDict
import torch 
from torch import nn 
import torch.nn.functional as F 

import pytorch_lightning as pl 
from sklearn.metrics import confusion_matrix
from tqdm import tqdm 
import itertools
import numpy as np 
import random 
import os 

from training import losses 
from training import basic_models
from data import data

class ReasoningModel(pl.LightningModule):
    """ Pytorch Lightning Module to train the reasoning layer of the Neuroplytorch model. Uses an LSTM neural network for a differentiable reasoning layer.

    """
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64, use_relu: bool = False, 
                annealing_step: int=1, loss_str: str = 'MSELoss', lr: float = 0.001, name: str=''):
        """ 
        
        :param input_size: Size of the input data (each element of the sequence).
        :type input_size: int
        :param output_size: Size of the output, for the case of Neuroplex this would be the number of complex events, resulting in an output vector of size output_size.
        :type output_size: int
        :param hidden_size: LSTM hidden layer size. Defaults to 64.
        :type hidden_size: int, optional
        :param use_relu: Use ReLU on output or not. This is for use with Evidential Deep Learning loss functions. Defaults to False. 
        :type use_relu: bool, optional
        :param annealing_step: Annealing step parameter for Evidential Deep Learning loss functions. Defaults to 1.
        :type annealing_step: int, optional
        :param loss_str: Loss function name to use for training. Defaults to 'MSELoss'.
        :type loss_str: str, optional
        :param lr: Learning rate for optimizer. Defaults to 0.001.
        :type lr: float, optional
        :param name: Separates the checkpoint weights, so experiment name goes here. Defalts to ''.
        :type name: str, optional
        """
        super().__init__()
        
        self.input_size = input_size 
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.use_relu = use_relu 
        self.annealing_step = annealing_step
        self.loss_str = loss_str 
        self.lr = lr 
        self.name = name 
        self.default_device = losses.fetch_default_device() 

        # save hyperparams for logging (Tensorboard)
        #self.save_hyperparameters()

        self.model = basic_models.BasicLSTM(self.input_size, self.output_size, hidden_size=self.hidden_size, use_relu=self.use_relu)

        self.loss_fct = losses.get_loss_fct(loss_str)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Model forward pass

        :param x: Model input
        :type x: torch.tensor

        :return: Model output
        :rtype: torch.tensor
        """
        return self.model(x) 

    def calculate_loss(self, yPred: torch.tensor, yTrue: torch.tensor) -> torch.tensor:
        """ Calculate loss of forward pass using loss function provided on model initialisation

        :param yPred: Output of model 
        :type yPred: torch.tensor
        :param yTrue: Ground truth label
        :type yTrue: torch.tensor

        :return: Loss of yPred with yTrue as ground truth
        :type: torch.tensor
        """
        if self.loss_str=='MSELoss': return F.mse_loss(yPred, yTrue)
        elif self.loss_str=='CrossEntropyLoss': return F.cross_entropy(yPred, yTrue)
        elif self.loss_str in losses.get_edl_losses(): return losses.edl_log_loss(yPred, yTrue, self.current_epoch, self.output_size, self.annealing_step)

    def calculate_accuracy(self, yPred: torch.tensor, yTrue: torch.tensor) -> torch.tensor:
        # accuracy calculated as output vector has to exactly match label vector (e.g. [0,1,0,0]==[0,1,0,0] is correct (1.0), [0,0,0,0]==[0,1,0,0] is incorrect (0.0), rather than 0.75)
        """ Calculate accuracy of forward pass. Reasoning model accuracy for a single instance is either 1.0 or 0.0, and must have both vectors match perfectly to be correct, otherwise is 0.0, regardless of whether some elements of each vector match.

        :param yPred: Output of model 
        :type yPred: torch.tensor
        :param yTrue: Ground truth label
        :type yTrue: torch.tensor

        :return: Mean accuracy of yPred with yTrue as ground truth
        :type: torch.tensor
        """
        return torch.mean(torch.all(torch.round(yPred)==yTrue, dim=1).float())

    def configure_optimizers(self):
        # TODO: allow optimizer to be hyperparameter
        """ Configure the optimizer (Adam) for training using a learning rate provided on model initialisation

        :return: Optimizer set to model parameters
        """
        # TODO: maybe make optimizer a hyperparam? 
        return torch.optim.Adam(self.parameters(), lr=self.lr) 
        
    def evaluate(self, batch: torch.tensor, batch_idx: int, stage:str=None) -> dict:
        """ Forward pass a batch (used for train, val and test batches). Each step/epoch is logged with metrics loss and accuracy using Tensorboard.

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        :param stage: Name of the current stage i.e. train, val or test. Defaults to None.
        :type stage: str, optional

        :return: Loss and accuracy as keys
        :rtype: dict
        """
        
        x, label = batch['x'], batch['label'] 
        x = x.to(self.default_device).float()
        label = label.to(self.default_device).float()

        y_pred = self(x) 
        loss = self.calculate_loss(y_pred, label)
        acc = self.calculate_accuracy(y_pred, label)

        # log the accuracy and loss for this stage, on each step and on each epoch, update the progress bar
        if stage:
            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{stage}_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'accuracy': acc}

    def training_step(self, batch: torch.tensor, batch_idx: int) -> dict:
        """ On training step, call self.evaluate (forward pass) and return

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        
        :return: Output from self.evaluate
        :rtype: dict
        """
        return self.evaluate(batch, batch_idx, 'train')

    def validation_step(self, batch: torch.tensor, batch_idx: int) -> dict:
        """ On validation step, call self.evaluate (forward pass) and return

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        
        :return: Output from self.evaluate
        :rtype: dict
        """
        return self.evaluate(batch, batch_idx, 'val')

    def test_step(self, batch: torch.tensor, batch_idx: int) -> dict:
        """ On test step, call self.evaluate (forward pass) and return

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        
        :return: Output from self.evaluate
        :rtype: dict
        """
        return self.evaluate(batch, batch_idx, 'test')

    def save_weights(self, cls):
        """ Save the model weights only using pure Pytorch implementation of saving weights.
        """
        torch.save(self.model.state_dict(), cls)

    # TODO: perhaps save the best only? 
    # save the weights at each epoch end 
    def on_train_epoch_end(self):
        self.save_weights(f'curr_tmp_{self.name}_reasoning_model.pt')
        
class PerceptionWindow(pl.LightningModule):
    """ Pytorch Lightning Module wrapping the LeNet pure Pytorch model for training with the EndToEndDataModule or EndToEndNoTestDataModule using Pytorch Lightning. Uses the intermediate label
    from the dataset to train the model on windowed data, similar to how the pereception layer is trained in the Neuroplytorch model but excluding the reasoning layer.
    """
    def __init__(self, perception_model: nn.Module, window_size: int, num_primitive_events: int, loss_str: str='MSELoss', lr: float=0.001):
        """

        :param perception_model: The perception model to use on the windowed data
        :type perception_mode: torch.nn.Module
        :param window_size: Size of the window of primitive events. 
        :type window_size: int
        :param num_primitive_events: Number of primitive events i.e. size of one hot primitive event vectors. 
        :type num_primitive_events: int
        :param loss_str: Loss function name to use for training. Defaults to 'MSELoss'.
        :type loss_str: str, optional
        :param lr: Learning rate for optimizer. Defaults to 0.001.
        :type lr: float, optional
        """

        super().__init__() 
        self.model = perception_model

        self.loss_str = loss_str 
        self.lr = lr 
        self.default_device = losses.fetch_default_device() 

        self.window_size = window_size 
        self.num_primitive_events = num_primitive_events

        #self.save_hyperparameters()

        self.loss_fct = losses.get_loss_fct(loss_str)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Model forward pass. Takes the windowed input data from x, and for each instance in window, the model is inferenced. The batch and window axis are swapped so each model forward pass is over the batch of element *i* in the window. 

        :param x: Model input
        :type x: torch.tensor

        :return: Model output
        :rtype: torch.tensor
        """
        x = torch.swapaxes(x, 0, 1)
        outs = []
        for xx in x: outs.append(self.model(xx))
        outs = torch.stack(outs)
        outs = torch.swapaxes(outs, 0, 1)
        return outs 
        

    def calculate_loss(self, yPred: torch.tensor, yTrue: torch.tensor) -> torch.tensor:
        """ Calculate loss of model output with respect to ground truth and loss function provided on model initialisation

        :param yPred: Output of model 
        :type yPred: torch.tensor
        :param yTrue: Ground truth label
        :type yTrue: torch.tensor

        :return: Loss of yPred with yTrue as ground truth
        :type: torch.tensor
        """
        if self.loss_str=='MSELoss': return F.mse_loss(yPred, yTrue)
        elif self.loss_str=='CrossEntropyLoss': return F.cross_entropy(yPred, yTrue)
        elif self.loss_str in losses.get_edl_losses(): return losses.edl_log_loss(yPred, yTrue, self.current_epoch, self.output_size, self.annealing_step)


    def calculate_accuracy(self, yPred: torch.tensor, yTrue: torch.tensor) -> torch.tensor:
        """ Calculate accuracy of forward pass. 

        :param yPred: Output of model 
        :type yPred: torch.tensor
        :param yTrue: Ground truth label
        :type yTrue: torch.tensor

        :return: Mean accuracy of yPred with yTrue as ground truth
        :type: torch.tensor
        """
        return torch.mean((torch.argmax(yPred, dim=2)==torch.argmax(yTrue, dim=2)).float())

    def configure_optimizers(self):
        """ Configure the optimizer (Adam) for training using a learning rate provided on model initialisation

        :return: Optimizer set to model parameters
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def evaluate(self, batch: torch.tensor, batch_idx: int, stage:str=None) -> dict:
        """ Forward pass a batch (used for train, val and test batches). Each step/epoch is logged with metrics loss and accuracy using Tensorboard.

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        :param stage: Name of the current stage i.e. train, val or test. Defaults to None.
        :type stage: str, optional

        :return: Loss and accuracy as keys
        :rtype: dict
        """
        x, label = batch['x'], batch['label_intermediate'] 

        y_pred = self(x) 
        loss = self.calculate_loss(y_pred, label)
        acc = self.calculate_accuracy(y_pred, label)

        if stage:
            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{stage}_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'accuracy': acc}

    def training_step(self, batch: torch.tensor, batch_idx: int) -> dict:
        """ On training step, call self.evaluate (forward pass) and return

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        
        :return: Output from self.evaluate
        :rtype: dict
        """
        return self.evaluate(batch, batch_idx, 'train')

    def validation_step(self, batch: torch.tensor, batch_idx: int) -> dict:
        """ On validation step, call self.evaluate (forward pass) and return

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        
        :return: Output from self.evaluate
        :rtype: dict
        """
        return self.evaluate(batch, batch_idx, 'val')

    def test_step(self, batch: torch.tensor, batch_idx: int) -> dict:
        """ On test step, call self.evaluate (forward pass) and return

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        
        :return: Output from self.evaluate
        :rtype: dict
        """
        return self.evaluate(batch, batch_idx, 'test')

class Neuroplytorch(pl.LightningModule):
    """ Neuroplytorch model for end-to-end training. Takes the trained reasoning model of type class:`training.basic_models.BasicLSTM` and a perception model that inherits from torch.nn.Module to create the end-to-end model for inferencing from raw input data to complex event labels.

    """
    def __init__(self, reasoning_model: basic_models.BasicLSTM, perception_model: PerceptionWindow, window_size: int, num_primitive_events: int, loss_str: str, lr: float):
        """ 

        :param reasoning_model:
        :type reasoning_model: class:`training.basic_models.BasicLSTM`
        :param perception_model: The perception model to use on the windowed data
        :type perception_mode: class:`training.models.PerceptionWindow`
        :param window_size: Size of the window of primitive events. 
        :type window_size: int
        :param num_primitive_events: Number of primitive events i.e. size of one hot primitive event vectors.
        :type num_primitive_events: int
        :param loss_str: Loss function name to use for training. 
        :type loss_str: str
        :param lr: Learning rate for optimizer. 
        :type lr: float
        """
        super().__init__() 
        
        self.perception_model = perception_model
        self.reasoning_model = reasoning_model

        self.loss_str = loss_str 
        self.default_device = losses.fetch_default_device() 

        self.window_size = window_size 
        self.num_primitive_events = num_primitive_events

        #self.save_hyperparameters()

        self.loss_fct = losses.get_loss_fct(loss_str)

        self.lr = lr 

        # Ensure frozen reasoning layer and unfrozen perception layer 
        for param in self.reasoning_model.parameters(): param.requires_grad = False 
        for param in self.perception_model.parameters(): param.requires_grad = True

        # Used for keeping track of epoch-level complex accuracies and epoch-level predictions for confusion matrix
        self.epoch_complex_accuracies = {'train': [], 'val': [], 'test': []} 
        self.epoch_ypreds = {'train': [], 'val': [], 'test': []} 
        self.epoch_ytrues = {'train': [], 'val': [], 'test': []} 

    def forward(self, x: torch.tensor) -> tuple:
        """
        Model forward pass. First passes through the perception layer. which outputs a window of inferences, which is then passed through the reasoning layer to inference the complex event label.

        :param x: Model input
        :type x: torch.tensor

        :return: Tuple of intermediate_output, reasoning_output
        :rtype: tuple
        """
        intermediate = self.perception_model(x) 
        reasoning_out = self.reasoning_model(intermediate)
        return intermediate, reasoning_out
        
    def calculate_loss(self, yPred: torch.tensor, yTrue: torch.tensor) -> torch.tensor:
        """ Calculate loss of model output with respect to ground truth and loss function provided on model initialisation

        :param yPred: Output of model 
        :type yPred: torch.tensor
        :param yTrue: Ground truth label
        :type yTrue: torch.tensor

        :return: Loss of yPred with yTrue as ground truth
        :type: torch.tensor
        """
        if self.loss_str=='MSELoss': return F.mse_loss(yPred, yTrue)
        elif self.loss_str=='CrossEntropyLoss': return F.cross_entropy(yPred, yTrue)
        elif self.loss_str in losses.get_edl_losses(): return losses.edl_log_loss(yPred, yTrue, self.current_epoch, self.output_size, self.annealing_step)
    
    def calculate_accuracy(self, yPred: torch.tensor, yTrue: torch.tensor) -> torch.tensor:
        """ Calculate accuracy of forward pass. Neuroplytorch model accuracy for a single instance is either 1.0 or 0.0, and must have both vectors match perfectly to be correct, otherwise is 0.0, regardless of whether some elements of each vector match.

        :param yPred: Output of model 
        :type yPred: torch.tensor
        :param yTrue: Ground truth label
        :type yTrue: torch.tensor

        :return: Mean accuracy of yPred with yTrue as ground truth
        :type: torch.tensor
        """
        return torch.mean(torch.all(torch.round(yPred)==yTrue, dim=1).float())

    def calculate_complex_accuracy(self, yPred: torch.tensor, yTrue: torch.tensor) -> torch.tensor:
        """ Calculate accuracy of forward pass, but only considering instances where either the inference or ground truth are not 'zero-complex-event windows', i.e. the accuracy of instances where the prediction or ground truth has at least one complex event. This gives a better representation of model accuracy given the usual very large bias towards windows with no complex events occuring.

        :param yPred: Output of model 
        :type yPred: torch.tensor
        :param yTrue: Ground truth label
        :type yTrue: torch.tensor

        :return: Mean accuracy of yPred with yTrue as ground truth
        :type: torch.tensor
        """
        a = torch.all(torch.round(yPred)==yTrue, dim=1).float()
        b = torch.sum(yTrue, dim=1).bool()
        c = torch.sum(torch.round(yPred), dim=1).bool()
        d = torch.logical_or(b, c)

        return a[d]

    def calculate_intermediate_accuracy(self, yPredIntermediate: torch.tensor, yTrueIntermediate: torch.tensor) -> torch.tensor:
        """ Calculate accuracy of the intermediate output in forward pass, i.e. the accuracy of the perception layer relative to the intermediate labels. Output of model and labels are windowed.

        :param yPredIntermediate: Output of model 
        :type yPredIntermediate: torch.tensor
        :param yTrueIntermediate: Ground truth label
        :type yTrueIntermediate: torch.tensor

        :return: Mean accuracy of yPredIntermediate with yTrueIntermediate as ground truth
        :type: torch.tensor
        """
        a = (torch.argmax(yPredIntermediate, dim=2)==torch.argmax(yTrueIntermediate, dim=2)).float() 
        return torch.mean(a)

    def configure_optimizers(self):
        """ Configure the optimizer (Adam) for training using a learning rate provided on model initialisation

        :return: Optimizer set to model parameters
        """
        return torch.optim.Adam(self.perception_model.parameters(), lr=self.lr)
    
    def evaluate(self, batch: torch.tensor, batch_idx: int, stage:str=None) -> dict:
        """ Forward pass a batch (used for train, val and test batches). Each step/epoch is logged with metrics loss, accuracy and inter_accuracy using Tensorboard.

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        :param stage: Name of the current stage i.e. train, val or test. Defaults to None.
        :type stage: str, optional

        :return: Loss and accuracy as keys
        :rtype: dict
        """
        x, label, label_intermediate = batch['x'], batch['label'], batch['label_intermediate']

        intermediate_out, reasoning_out = self(x) 
        loss = self.calculate_loss(reasoning_out, label)
        acc = self.calculate_accuracy(reasoning_out, label)
        self.epoch_complex_accuracies[stage].append(self.calculate_complex_accuracy(reasoning_out, label))
        intermediate_acc = self.calculate_intermediate_accuracy(intermediate_out, label_intermediate)

        # CONFUSION MATRIX CODE 
        yPred = torch.argmax(torch.round(reasoning_out), dim=1).float()
        yPred_add = torch.sum(torch.round(reasoning_out), dim=1).bool().float()
        yPred += yPred_add 

        yTrue = torch.argmax(label, dim=1).float()
        yTrue_add = torch.sum(label, dim=1).bool().float()
        yTrue += yTrue_add

        self.epoch_ypreds[stage].append(yPred)
        self.epoch_ytrues[stage].append(yTrue)

        if stage:
            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{stage}_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{stage}_inter_accuracy", intermediate_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'accuracy': acc}

    def training_step(self, batch: torch.tensor, batch_idx: int) -> dict:
        """ On training step, call self.evaluate (forward pass) and return

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        
        :return: Output from self.evaluate
        :rtype: dict
        """
        return self.evaluate(batch, batch_idx, 'train')

    def validation_step(self, batch: torch.tensor, batch_idx: int) -> dict:
        """ On validation step, call self.evaluate (forward pass) and return

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        
        :return: Output from self.evaluate
        :rtype: dict
        """
        return self.evaluate(batch, batch_idx, 'val')

    def test_step(self, batch: torch.tensor, batch_idx: int) -> dict:
        """ On test step, call self.evaluate (forward pass) and return

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        
        :return: Output from self.evaluate
        :rtype: dict
        """
        return self.evaluate(batch, batch_idx, 'test')

    def training_epoch_end(self, outputs) -> None:
        """ On training epoch end, calculate and log the complex accuracy across the whole epoch. Since the bias towards windows with no complex events occuring is large, some batches may be entirely 'zero-window' instances, and so the calculation has to occur at the end of the epoch rather than per-batch.

        """
        self.log('train_complex_accuracy', torch.mean(torch.concat(self.epoch_complex_accuracies['train'])), on_epoch=True, logger=True, prog_bar=False, on_step=False)

    def validation_epoch_end(self, outputs) -> None:
        """ On validation epoch end, calculate and log the complex accuracy across the whole epoch. Also create a confusion matrix from the validation inferences across the epoch and log.
        
        """
        self.log('val_complex_accuracy', torch.mean(torch.concat(self.epoch_complex_accuracies['val'])), on_epoch=True, logger=True, prog_bar=False, on_step=False)
        
        ytrue = torch.cat(self.epoch_ytrues['val'])
        ypred = torch.cat(self.epoch_ypreds['val']) 
        print(confusion_matrix(ytrue.cpu().detach().numpy(), ypred.cpu().detach().numpy()))
        self.epoch_ypreds = {'train': [], 'val': [], 'test': []} 
        self.epoch_ytrues = {'train': [], 'val': [], 'test': []} 
    
    def test_epoch_end(self, outputs) -> None:
        """ On training epoch end, calculate and log the complex accuracy across the whole epoch. 

        """
        self.log('test_complex_accuracy', torch.mean(torch.concat(self.epoch_complex_accuracies['test'])), on_epoch=True, logger=True, prog_bar=False, on_step=False)

    def save_model(self, dir):
        if not os.path.isdir(dir): os.mkdir(dir)
        torch.save(self.reasoning_model.state_dict(), dir+"/reasoning_model.pt")
        torch.save(self.perception_model.model.state_dict(), dir+"/perception_model.pt")

    def load_model(self, dir):
        self.reasoning_model.load_state_dict(torch.load(dir+"/reasoning_model.pt", map_location='cpu')) 
        self.perception_model.model.load_state_dict(torch.load(dir+"/perception_model.pt", map_location='cpu'))
    

# # CUSTOM MODULES FOR PRETRAINING PERCEPTION MODELS
class MNISTModel(pl.LightningModule):
    """ Pytorch Lightning Module wrapping the LeNet pure Pytorch model for training with the MNISTDataModule using Pytorch Lightning
    """
    def __init__(self, loss_str: str, lr: float):
        """

        :param loss_str: Loss function name to use for training.
        :type loss_str: str
        :param lr: Learning rate for optimizer. 
        :type lr: float 
        """
        super().__init__()
        
        self.loss_str = loss_str 
        self.lr = lr 
        self.default_device = losses.fetch_default_device() 

        #self.save_hyperparameters()

        self.model = basic_models.LeNet(output_size=10)

        self.loss_fct = losses.get_loss_fct(loss_str)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Model forward pass

        :param x: Model input
        :type x: torch.tensor

        :return: Model output
        :rtype: torch.tensor
        """
        return self.model(x) 

    def calculate_loss(self, yPred: torch.tensor, yTrue: torch.tensor) -> torch.tensor:
        """ Calculate loss of model output with respect to ground truth and loss function provided on model initialisation

        :param yTrue: Ground truth label
        :type yTrue: torch.tensor
        :param yPred: Output of model 
        :type yPred: torch.tensor

        :return: Loss of yPred with yTrue as ground truth
        :type: torch.tensor
        """
        if self.loss_str=='MSELoss': return F.mse_loss(yPred, yTrue)
        elif self.loss_str=='CrossEntropyLoss': return F.cross_entropy(yPred, yTrue)
        elif self.loss_str in losses.get_edl_losses(): return losses.edl_log_loss(yPred, yTrue, self.current_epoch, self.output_size, self.annealing_step)

    def calculate_accuracy(self, yPred: torch.tensor, yTrue: torch.tensor) -> torch.tensor:
        """ Calculate accuracy of forward pass. 

        :param yPred: Output of model 
        :type yPred: torch.tensor
        :param yTrue: Ground truth label
        :type yTrue: torch.tensor

        :return: Mean accuracy of yPred with yTrue as ground truth
        :type: torch.tensor
        """
        return torch.mean((torch.argmax(yPred, dim=1)==yTrue).float())

    def configure_optimizers(self):
        """ Configure the optimizer (Adam) for training using a learning rate provided on model initialisation

        :return: Optimizer set to model parameters
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr) 

    def evaluate(self, batch: torch.tensor, batch_idx: int, stage:str=None) -> dict:
        """ Forward pass a batch (used for train, val and test batches). Each step/epoch is logged with metrics loss and accuracy using Tensorboard.

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        :param stage: Name of the current stage i.e. train, val or test. Defaults to None.
        :type stage: str, optional

        :return: Loss and accuracy as keys
        :rtype: dict
        """
        x, label = batch
        y_pred = self(x) 

        loss = self.calculate_loss(y_pred, label)
        acc = self.calculate_accuracy(y_pred, label)

        if stage:
            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{stage}_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'accuracy': acc}

    def training_step(self, batch: torch.tensor, batch_idx: int) -> dict:
        """ On training step, call self.evaluate (forward pass) and return

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        
        :return: Output from self.evaluate
        :rtype: dict
        """
        return self.evaluate(batch, batch_idx, 'train')

    def validation_step(self, batch: torch.tensor, batch_idx: int) -> dict:
        """ On validation step, call self.evaluate (forward pass) and return

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        
        :return: Output from self.evaluate
        :rtype: dict
        """
        return self.evaluate(batch, batch_idx, 'val')

    def test_step(self, batch: torch.tensor, batch_idx: int) -> dict:
        """ On test step, call self.evaluate (forward pass) and return

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        
        :return: Output from self.evaluate
        :rtype: dict
        """
        return self.evaluate(batch, batch_idx, 'test')

class EMNISTModel(pl.LightningModule):
    """ Pytorch Lightning Module wrapping the LeNet pure Pytorch model for training with the EMNISTDataModule using Pytorch Lightning
    """
    def __init__(self, loss_str: str, lr: float):
        """

        :param loss_str: Loss function name to use for training.
        :type loss_str: str
        :param lr: Learning rate for optimizer. 
        :type lr: float 
        """
        super().__init__()
        
        self.loss_str = loss_str 
        self.lr = lr 
        self.default_device = losses.fetch_default_device() 

        #self.save_hyperparameters()

        self.model = basic_models.LeNet(output_size=26)

        self.loss_fct = losses.get_loss_fct(loss_str)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Model forward pass

        :param x: Model input
        :type x: torch.tensor

        :return: Model output
        :rtype: torch.tensor
        """
        return self.model(x) 

    def calculate_loss(self, yPred: torch.tensor, yTrue: torch.tensor) -> torch.tensor:
        """ Calculate loss of model output with respect to ground truth and loss function provided on model initialisation

        :param yTrue: Ground truth label
        :type yTrue: torch.tensor
        :param yPred: Output of model 
        :type yPred: torch.tensor

        :return: Loss of yPred with yTrue as ground truth
        :type: torch.tensor
        """
        if self.loss_str=='MSELoss': return F.mse_loss(yPred, yTrue)
        elif self.loss_str=='CrossEntropyLoss': return F.cross_entropy(yPred, yTrue)
        elif self.loss_str in losses.get_edl_losses(): return losses.edl_log_loss(yPred, yTrue, self.current_epoch, self.output_size, self.annealing_step)

    def calculate_accuracy(self, yPred: torch.tensor, yTrue: torch.tensor) -> torch.tensor:
        """ Calculate accuracy of forward pass. 

        :param yPred: Output of model 
        :type yPred: torch.tensor
        :param yTrue: Ground truth label
        :type yTrue: torch.tensor

        :return: Mean accuracy of yPred with yTrue as ground truth
        :type: torch.tensor
        """
        return torch.mean((torch.argmax(yPred, dim=1)==yTrue).float())

    def configure_optimizers(self):
        """ Configure the optimizer (Adam) for training using a learning rate provided on model initialisation

        :return: Optimizer set to model parameters
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr) 

    def evaluate(self, batch: torch.tensor, batch_idx: int, stage:str=None) -> dict:
        """ Forward pass a batch (used for train, val and test batches). Each step/epoch is logged with metrics loss and accuracy using Tensorboard.

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        :param stage: Name of the current stage i.e. train, val or test. Defaults to None.
        :type stage: str, optional

        :return: Loss and accuracy as keys
        :rtype: dict
        """
        x, label = batch
        y_pred = self(x) 

        loss = self.calculate_loss(y_pred, label)
        acc = self.calculate_accuracy(y_pred, label)

        if stage:
            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{stage}_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'accuracy': acc}

    def training_step(self, batch: torch.tensor, batch_idx: int) -> dict:
        """ On training step, call self.evaluate (forward pass) and return

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        
        :return: Output from self.evaluate
        :rtype: dict
        """
        return self.evaluate(batch, batch_idx, 'train')

    def validation_step(self, batch: torch.tensor, batch_idx: int) -> dict:
        """ On validation step, call self.evaluate (forward pass) and return

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        
        :return: Output from self.evaluate
        :rtype: dict
        """
        return self.evaluate(batch, batch_idx, 'val')

    def test_step(self, batch: torch.tensor, batch_idx: int) -> dict:
        """ On test step, call self.evaluate (forward pass) and return

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        
        :return: Output from self.evaluate
        :rtype: dict
        """
        return self.evaluate(batch, batch_idx, 'test')

class UrbanModel(pl.LightningModule):
    """ Pytorch Lightning Module wrapping the LeNet pure Pytorch model for training with the EMNISTDataModule using Pytorch Lightning
    """
    def __init__(self, loss_str: str, lr: float):
        """

        :param loss_str: Loss function name to use for training. 
        :type loss_str: str
        :param lr: Learning rate for optimizer. 
        :type lr: float 
        """
        super().__init__()
        
        self.loss_str = loss_str 
        self.lr = lr 
        self.default_device = losses.fetch_default_device() 

        #self.save_hyperparameters()

        self.model = basic_models.VGGish(output_size=10)

        self.loss_fct = losses.get_loss_fct(loss_str)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Model forward pass

        :param x: Model input
        :type x: torch.tensor

        :return: Model output
        :rtype: torch.tensor
        """
        return self.model(x) 

    def calculate_loss(self, yPred: torch.tensor, yTrue: torch.tensor) -> torch.tensor:
        """ Calculate loss of model output with respect to ground truth and loss function provided on model initialisation

        :param yTrue: Ground truth label
        :type yTrue: torch.tensor
        :param yPred: Output of model 
        :type yPred: torch.tensor

        :return: Loss of yPred with yTrue as ground truth
        :type: torch.tensor
        """
        if self.loss_str=='MSELoss': return F.mse_loss(yPred, yTrue)
        elif self.loss_str=='CrossEntropyLoss': return F.cross_entropy(yPred, yTrue)

    def calculate_accuracy(self, yPred: torch.tensor, yTrue: torch.tensor) -> torch.tensor:
        """ Calculate accuracy of forward pass. 

        :param yPred: Output of model 
        :type yPred: torch.tensor
        :param yTrue: Ground truth label
        :type yTrue: torch.tensor

        :return: Mean accuracy of yPred with yTrue as ground truth
        :type: torch.tensor
        """
        return torch.mean((torch.argmax(yPred, dim=1)==yTrue).float())

    def configure_optimizers(self):
        """ Configure the optimizer (Adam) for training using a learning rate provided on model initialisation

        :return: Optimizer set to model parameters
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr) 

    def evaluate(self, batch: torch.tensor, batch_idx: int, stage:str=None) -> dict:
        """ Forward pass a batch (used for train, val and test batches). Each step/epoch is logged with metrics loss and accuracy using Tensorboard.

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        :param stage: Name of the current stage i.e. train, val or test. Defaults to None.
        :type stage: str, optional

        :return: Loss and accuracy as keys
        :rtype: dict
        """
        x, label = batch
        y_pred = self(x) 

        loss = self.calculate_loss(y_pred, label)
        acc = self.calculate_accuracy(y_pred, label)

        if stage:
            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{stage}_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'accuracy': acc}

    def training_step(self, batch: torch.tensor, batch_idx: int) -> dict:
        """ On training step, call self.evaluate (forward pass) and return

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        
        :return: Output from self.evaluate
        :rtype: dict
        """
        return self.evaluate(batch, batch_idx, 'train')

    def validation_step(self, batch: torch.tensor, batch_idx: int) -> dict:
        """ On validation step, call self.evaluate (forward pass) and return

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        
        :return: Output from self.evaluate
        :rtype: dict
        """
        return self.evaluate(batch, batch_idx, 'val')

    def test_step(self, batch: torch.tensor, batch_idx: int) -> dict:
        """ On test step, call self.evaluate (forward pass) and return

        :param batch: Input data as batch
        :type batch: torch.tensor
        :param batch_idx: Unused but needed in overloading
        :type batch_idx: int
        
        :return: Output from self.evaluate
        :rtype: dict
        """
        return self.evaluate(batch, batch_idx, 'test')


#TODO: on train end, val end and test end (epoch) - log/print confusion matrix 




def get_model(model_name_str: str) -> pl.LightningModule:
    """ Get the Pytorch Lightning module associated with the string given, which is defined in the config file
    :param model_name_str: Name of the module to return
    :type model_name_str: str

    :return: The Pytorch Lightning module
    :rtype: pytorch_lightning.LightningModule
    """

    if model_name_str=='MNISTModel': return MNISTModel
    elif model_name_str=='EMNISTModel': return EMNISTModel
    elif model_name_str=='UrbanModel': return UrbanModel

def check_reasoning_logic(reasoning_model: nn.Module, ce_fsm_list: list, ce_time_list: list, num_primitive_events: int, window_size: int):
    """ Check the logic of the reasoning model through all possible permutations of windowed primitive events to ensure that this layer will correctly label each instance.

    :param reasoning_model: The reasoning model to check.
    :type reasoning_model: nn.Module (or a class that inherits nn.Module, e.g. pytorch_lightning.LightningModule)
    :param ce_fsm_list: Pattern of primitive events for each complex event.
    :type ce_fsm_list: list
    :param ce_time_list: Temporal metadata pattern for each complex event
    :type ce_time_list: list
    :param num_primitive_events: Number of primitive events i.e. size of one hot primitive event vectors. 
    :type num_primitive_events: int
    :param window_size: Size of the window of primitive events. 
    :type window_size: int
    """

    default_device = losses.fetch_default_device()
    reasoning_model.to(default_device)
    verify_result = True 
    uniq_event = list(range((num_primitive_events))) 
    total_len = len(uniq_event)**window_size
    nums = 0
    product_list = itertools.product(uniq_event, repeat=window_size)
    for i in tqdm(product_list, total=total_len):
        event_stream = np.array(list(i))  
        event_feature = np.zeros([window_size, len(uniq_event)])
        event_feature[np.arange(window_size), event_stream] = 1
        complex_event = data.get_complex_label(torch.tensor(event_feature), ce_fsm_list, ce_time_list, False).to(default_device)

        event_feature = torch.tensor([event_feature]).float().to(default_device)
        yPred = reasoning_model(event_feature)
        
        logic_match = (torch.round(yPred)==complex_event).all().item()
        
        if not logic_match:
            verify_result = False 
            print(i) 
            break 
    
    if verify_result: print("Successful, all possible permutations are correct")
    else: print("Above window failed logical check")
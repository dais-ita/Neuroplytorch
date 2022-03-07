from turtle import forward
import torch 
from torch import nn 
import torch.nn.functional as F 

class BasicLSTM(nn.Module):
    """Pure Pytorch implementation of a very simple LSTM model with a single layer followed by a dense classification layer
    """
    def __init__(self, input_size: int = 10, output_size: int = 4, **kwargs):
        """
        :param input_size: Size of the input data (each element of the sequence). Defaults to 10.
        :type input_size: int, optional
        :param output_size: Size of the output, for the case of Neuroplex this would be the number of complex events, resulting in an output vector of size output_size. Defaults to 4.
        :type output_size: int, optional
        :param hidden_size: LSTM hidden layer size. Defaults to 64.
        :type hidden_size: int, optional
        :param use_relu: Use ReLU on output or not. This is for use with Evidential Deep Learning loss functions. Defaults to False. 
        :type use_relu: bool, optional
        """
        super(BasicLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = kwargs.get('hidden_size', 64)
        self.use_relu = kwargs.get('use_relu', False)

        self.lstm_layer = nn.LSTM(
            input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.out_layer = nn.Linear(self.hidden_size, self.output_size)

        nn.init.xavier_normal_(self.out_layer.weight)
        for layer_p in self.lstm_layer._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.normal_(self.lstm_layer.__getattr__(p), 0.0, 0.02)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """ LSTM forward pass function. If initialised model with use_relu as True, will return logits after ReLU function, otherwise pure logits.

        :param x: Model input
        :type x: torch.tensor

        :return: Output of model given x
        :rtype: torch.tensor
        """
        output, (h, c) = self.lstm_layer(x)
        #output = self.dropout_layer(output)
        logits = self.out_layer(output[:, -1, :])
        if self.use_relu: logits = F.relu(logits)

        return logits

class LeNet(nn.Module):
    """ 
    Pure Pytorch implementation of LeNet-5 from LeCun, Y., Bottou, L., Bengio, Y. and Haffner, P., 1998. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), pp.2278-2324.
    """
    def __init__(self, input_size: int = 1, output_size: int=10, **kwargs):
        """

        :param output_size: Numer of outputs/classes of the model. Defaults to 10.
        :type output_size: int, optional
        """
        super(LeNet, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=self.input_size, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=self.output_size),
        )


    def forward(self, x: torch.tensor) -> torch.tensor:
        """ LeNet forward pass function

        :param x: Model input
        :type x: torch.tensor

        :return: Output of model given x
        :rtype: torch.tensor
        """
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return probs

class VGGish(nn.Module):
    """ 
    Pure Pytorch implementation of LeNet-5 from LeCun, Y., Bottou, L., Bengio, Y. and Haffner, P., 1998. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), pp.2278-2324.
    """
    def __init__(self, input_size: int = 128, output_size: int = 10, **kwargs):
        """

        :param output_size: Numer of outputs/classes of the model. Defaults to 10.
        :type output_size: int, optional
        """
        super(VGGish, self).__init__() 

        self.input_size = input_size 
        self.output_size = output_size 

        self.net = nn.Sequential( 
            nn.Linear(in_features=self.input_size, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=80),
            nn.ReLU(),
            nn.Linear(in_features=80, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=25),
            nn.ReLU(),
            nn.Linear(in_features=25, out_features=self.output_size)
        ) 

    def forward(self, x: torch.tensor) -> torch.tensor: 
        """ VGGish forward pass function

        :param x: Model input
        :type x: torch.tensor

        :return: Output of model given x
        :rtype: torch.tensor
        """
        logits = self.net(x)
        probs = F.softmax(logits, dim=1)
        return probs 

def get_model(model_name_str: str) -> nn.Module:
    """ Get the pure Pytorch model associated with the string given, which is defined in the config file
    :param model_name_str: Name of the model to return
    :type model_name_str: str

    :return: The pure Pytorch model
    :rtype: nn.Module
    """
    
    if model_name_str=='LeNet': return LeNet
    elif model_name_str=='VGGish': return VGGish
    elif model_name_str=='BasicLSTM': return BasicLSTM 

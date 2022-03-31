.. _How To Page:

Getting Started
================================

This page will introduce you to the Neuroplytorch framework, and will show how to run, how to create new scenarios and introduce new datasets, as well as 
how to implement new neural network architectures for the perception layer. This framework is a Pytorch implementation of https://github.com/nesl/Neuroplex
with additions for general use across multiple scenarios, datasets and perception layer models. 

Running Neuroplytorch
---------------------

The Neuroplytorch framework can be run in one of three ways:

   *  Reasoning model training followed by end-to-end training 
   *  End-to-end training 
   *  Reasoning model logic check 

The first two in this list depend on whether a reasoning model has already been trained, which is saved to the models/ directory, i.e. a reasoning model 
will be loaded and training skipped if there is a reasoning model to be found, else it will be trained from scratch and saved for future runs. The logic
check is run if the \--\logic flag is given as a program argument, and will entirely skip any training and simply check the logic of the saved reasoning
model. 

WARNING: The logic check goes through every possible permutation of windowed simple events, and will take a long time to run fully, although it's possible
to run for a short while to somewhat ensure the logic is decently sound, as the function will break early if any input breaks the logic.

To run the framework, use the following command in a terminal of your choice:

.. code-block:: console

   python main.py

Arguments:
   * \--\name: str, Name of the file in configs/ directory to use, as well as the name of the models saved. Defaults to basic_neuro_experiment.
   * \--\logic: int, If 1 then check the logic of the loaded Reasoning model, else run end-to-end training. Defaults to 0 

Example terminal execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   python main.py --name basic_neuro_experiment --logic 0

This will read from the file configs/basic_neuro_experiment.yaml and run end-to-end training. It will train a reasoning model if one hasn't been trained already.
With a trained reasoning layer, the argument \--\logic 0 will ignore end-to-end training and will check the logic of the newly trained reasoning layer (see warning).

Config files 
------------ 

Config files in the YAML format are stored in the configs/ directory, and are used to separate experiments/scenarios/runs and provide hyperparameters for each.
As mentioned above, the \--\name flag will point to the name of the config file to run. The config file is the centrepiece to define not only the hyperparameters for 
training, but also the dataset and neural network architectures to use, so for example one config file may define the dataset as a stream from a security camera, with 
an object detection model for the perception layer, another may point to audio data and a VGGish model. 

The config file is logged to Tensorboard for each run, so alterations between experiments can then be compared. Future work may include introducing libraries such as 
Optuna for hyperparameter search.

Config file structure
~~~~~~~~~~~~~~~~~~~~~

An example config file can be found in configs/example.yaml, which defines the MNIST problem with arbitrary complex events.

The basic structure of the file can be seen below, with a more detailed explanation following. Note that the config file is parsed into a Python dict object, and so some of the keys shown here can be omitted if not needed,
while others must be included; to show this, keys with a \* can be omitted if they are not needed for particular experiment (as can the sub keys that follow). Each leaf node 
will have a value rather than continue the tree, which could be a numerical value, a string, list etc. which is defined in the detailed explanation.

.. code-block:: none

    .
    ├── DETAILS
    │   ├── NAME
    │   ├── DESCRIPTION
    ├── TRAINING
    │   ├── NUM_PRIMITIVE_EVENTS
    │   ├── WINDOW_SIZE
    │   ├── DATASET
    │       ├── NAME
    │       ├── TYPE
    │       ├── LOCATION
    │   ├── REASONING
    │       ├── LOSS_FUNCTION*
    │       ├── LEARNING_RATE*
    │       ├── EPOCHS
    │       ├── PARAMETERS*
    │           ├── DATASET 
    │           ├── MODEL
    │   ├── NEUROPLYTORCH 
    │       ├── LEARNING_RATE*
    │       ├── LOSS_FUNCTION*
    │       ├── EPOCHS
    │       ├── PARAMETERS*
    │           ├── DATASET
    │           ├── MODEL 
    │   ├── PERCEPTION 
    │       ├── MODEL 
    │       ├── PRETRAIN*
    │           ├── MODEL_MODULE 
    │           ├── DATA_MODULE 
    │           ├── LOSS_FUNCTION
    │           ├── LEARNING_RATE
    │           ├── PRETRAIN_PERCEPTION
    │           ├── PRETRAIN_EPOCHS
    │           ├── PARAMETERS*
    │               ├── DATASET 
    │               ├── MODEL
    ├── COMPLEX EVENTS
    │   ├── COMPLEX_EVENT_NAME
    │       ├── PATTERN
    │       ├── EVENTS_BETWEEN
    │       ├── MAX_TIME
    │   ├── COMPLEX_EVENT_NAME
    │       ├── PATTERN
    │       ├── EVENTS_BETWEEN
    │       ├── MAX_TIME
        .
        .
        .

Details
"""""""
Here is simply a place to define the name and details of the experiment, and is simply for the use of elaboration on the experiment, e.g. what the scenario is, what the complex events are looking for etc.

   *  NAME: The name of the experiment. (text)
   *  DESCRIPTION: Description of the experiment. (text)

.. _training params:

Training
""""""""

Here is where the hyperparameters for training models (reasoning, end-to-end and pretraining perception) is defined, and the following subheadings (Reasoning, Neuroplytorch, Perception) are child nodes of
this key. 

   *  NUM_PRIMITIVE_EVENTS: The number of possible primitive/simple events that can be classified by the perception layer, e.g. would be 10 for MNIST as there are 10 possible classes that can be 
      predicted. (int)
   *  WINDOW_SIZE: The size of the window to inference over using the perception layer, and so defines the window size for the reasoning model. Experiments show that this value must be at least 10.
      [TODO]: add further information on window size. (int)
   *  DATASET: Defines hyperparameters for the raw input dataset:

      -  NAME: Arbitrary name of the dataset, doesn't affect the framework in any way if using a parser, but used as a high-level description for the sake of readability. Is used by the framework only in the case of using a Pytorch Dataset (see below), in which case is used to fetch the correct dataset.
      -  TYPE: Defines the format of the data, which is used to determine the parser to use. See :ref:`Creating a new parser <new parser>` for further details. 'Pytorch Dataset' will tell the framework to use torchvision datasets (e.g. MNIST or EMNIST).
      -  LOCATION: the relative or absolute directory location of the dataset to be used, e.g. datasets/MNIST will point to the relative datasets/MNIST directory in the framework whereas /home/name/datasets/MNIST will be an absolute directory.
   
Parameters 
"""""""""""

For each of the following definitions of the model and dataset hyperparameters (Reasoning, Neuroplytorch and Perception), the optional arguments are held here, split by MODEL and DATASET. 
These are passed into the specified LightningModule or LightningDataModule if specified, otherwise the default values are used. (TODO: quick example here of a LightningModule optional arguments)

Reasoning & Neuroplytorch
""""""""""""""""""""""""""

The arguments listed here, while technically filled with default values if omitted, are strongly encouraged to be included in the config file, given they are the important hyperparameters for training.
Since the hyperparameters of the config file are saved to Tensorboard when run, this allows each experiment to keep a log of these values.

   *  LOSS_FUNCTION - The name of the loss function to use, e.g. CrossEntropyLoss, MSELoss, edl_mse_loss etc. (text)
   *  LEARNING_RATE - The learning rate to use during training. (float)
   *  EPOCHS - The maximum number of epochs to train for; the 'maximum' implies this may not be the number of epochs actually trained over, given early-stopping etc. (int)

Perception
"""""""""""

The arguments defined here mostly pertain to pretraining the perception layer, that is to train the perception model on it's own using the perception labels of the dataset.
This is also where the perception model architecture is defined.

   *  MODEL - The name of the model architecture to use, as from :ref:`Basic Models <api/basic_models:Basic Models module>`, e.g. LeNet, VGGish etc. (text)
   *  INPUT_SIZE - The input_size parameter passed to the perception model. (int)
   *  PRETRAIN - This can be entirely omitted in the case of no pretraining, but holds the necessary parameters for pretraining:

      -  MODEL_MODULE - The PytorchLightningModule name to be used (see :ref:`Models <api/models:Models module>`)
      -  DATA_MODULE - The PytorchLightningDataModule name to be used (see :ref:`Pytorch Lightning data modules <api/datasets:Pytorch Lightning data modules>`)
      -  LOSS_FUNCTION - Name of the loss function to use (see source for training.losses.get_loss_fct from :ref:`losses <api/losses:Losses module>`)
      -  LEARNING_RATE - The learning rate to use for training.
      -  PRETRAIN_PERCEPTION - Boolean whether to pretrain or not.
      -  PRETRAIN_EPOCHS - The number of epochs to train the model for.

Complex events
""""""""""""""

Here the complex events are defined. This is implemented as a list of complex event names, which are arbitrary descriptions of the complex event, e.g. worrying_siren,
each of which has defined the perception pattern (PATTERN), the minimum number of simple events that must occur between events in the pattern (EVENTS_BETWEEN), and while 
it has yet to be implemented, the maximum time between events in the pattern (MAX_TIME). The perception pattern defines the pattern of simple events, as a vector, that must occur for 
the complex event to be occuring, with the last simple event in the pattern needing to match the last simple event in the perception window. For the events between and max  
time definitions, these are also vectors that define the number of events and max time between events, and so must be of size N-1 if N is the size of the pattern vector.


.. _adding dataset:

Adding your own dataset 
------------------------

In this section, the method for adding a new dataset is explained. The dataset is loaded by means of a parser, so if the data format hasn't yet been implemented,
there is a section on the structure for :ref:`creating a new parser <new parser>`. The dataset is held in a directory in ./data/X, where X is the name of the dataset.
The data should be pre-split into train and test directories, such that the only files in these directories is the raw data. A metadata file should be located in
the dataset's root directory, the format of this file is dependent on the parser, but ideally should be .csv. The metadata file is used to match the data to a class;
for text this will be some ID provided in the data file to match text to a classification, but for audio and video, this will match the file name to a classification.

For example, a line in a text data file might read "#100# This is the body of text" where #100# is the ID of that data point, and in the metadata the line might read 
"100,1,positive", so ID 100 is classed as 1, which is positive, for audio this might be "filename,1,positive". The exact structure of this can differ depending on the 
parser implementation, so see the :ref:`Parser API <api/parser:Parser module>` for the exact implementation based on the file format used.

As mentioned in :ref:`training parameters <training params>`, the file format of the dataset being used is defined in the config file, which the framework will use 
to select the appropriate parser. 

Implementing new models 
------------------------ 

Perception architectures are defined in :ref:`Basic Models <api/basic_models:Basic Models module>`, and are implemented as pure Pytorch models. The framework is designed
in such a way as to be 'plug-and-play' for end-to-end training, meaning as long as the perception architecture is defined in :ref:`Basic Models <api/basic_models:Basic Models module>`,
the if-else string-to-object statement is defined in :ref:`get model <api/basic_models:Methods>`, and the name of the model defined in the config file, then the framework 
will be able to train end-to-end with this perception architecture, with no other steps necessary. 

If pretraining is required, then a Pytorch Lightning module must be defined, see :ref:`Pretraining the perception layer <pretrain>`.

.. _new parser:

Creating a new parser 
---------------------

The parser is defined in :ref:`Parser module <api/parser:Parser module>`, and is responsible for loading the defined dataset from ./data into train and test sets. Using this 
API documentation as a reference, a new parser for a file format that hasn't yet been covered by this framework can be defined here. The if-else string-to-object statement is defined in
:ref:`fetch_perception_data_local <api/datasets:Methods>`, and so an if-clause should be added here to match a string name to parser method. It is recommended to read 
the above :ref:`Adding your own dataset <adding dataset>` section to get an understanding of how the data is stored, and so how to parse the data.

The format of the data to be used in defined in the config file (see :ref:`training params`), and currently the implementation only supports audio_wav for use with the parser functions. 
As more parsers are implemented, the number of formats available will be increased (as mentioned in the above if-clause), and should follow a similar standard, i.e. data type, underscore, data format.
Further examples may be video_mp4, or text_txt etc. for consistency. 

.. _pretrain:

Pretraining the perception layer
---------------------------------

This section only applies in the case of pretraining the perception layer, which can either be useful for training the perception layer on its own to see the performance of 
the implemented architecture with the chosen dataset, or to train the perception layer before end-to-end training. 


Loading perception weights 
---------------------------

Yet to be implemented. Will load pretrained weights for a perception model.

Zero-window regularisation
---------------------------

Yet to be implemented. Will tackle the issue of complex event class bias, i.e. the 'zero window' label (no complex events) outweighs all other labels (usually),
and so regularisation of this will need to be implemented in order to tackle this issue; especially apparent with EMNIST given the number of intermediate classes.

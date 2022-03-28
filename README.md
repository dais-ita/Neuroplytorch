# Neuroplytorch


## Demo

A video of the demo can be found in demo_outs/demo_video.mp4. The demo can be run from demo.py, and the optional arguments can be found in this file. The argument --mic will dictate the demo to run: 
<ul>
    <li> 0: will run through the audio specified by --file and after inferencing will display as a matplotlib plot the time series of the inferences.
    <li> 1: stream from a hardware microphone and display in real-time the inference of the model to the terminal.
    <li> 2: will stream audio defined by the --file argument in real-time while displaying the inference of the model to the terminal as above. This emulates a microphone
        without the background noise induced.
</ul>

The dataset for this demo can be downloaded from https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz - the audio files are split into 8 folds, for this demo we moved the files from folds 9 & 10 into the test folder, the other folds into the train folder. The metadata file is in it's own folder (UrbanSound8K/metadata) when downloaded, so this is moved to the root directory, as described in the documentation (under Adding your own dataset).
# aircraft_classification
> Code for training and running a classifier for aircraft.
> Based on FGVCA aircraft data (see https://arxiv.org/pdf/1306.5151.pdf), though this code allows one to use a subset

## Installation
pip install -q git+https://github.com/jme45/aircraft_classification.git

## Usage example
```python
from aircraft_classification import fit_model

results = fit_model.fit_model('vit_b_16',
                              'CIVILIAN_JETS',
                              2,
                              'runs',
                              )
```

This will save a model which can be loaded and then make predictions like this:
```python
import classifiers

vit = classifiers.AircraftClassifier('vit_b_16',
                                     'CIVILIAN_JETS',
                                     True,
                                     <path_to_pytorch_file>)

vit.predict(pil_image)
```

It's also possible to run the code from the command line.
```bash
./run -m 'vit_b_16' -a 'CIVILIAN_JETS' -n 2 -p 'runs'
```

## Approach
This code uses transfer learning, as I wouldn't have the computational resources
to train an entire computer vision model. It loads one of several model with
pretrained weights, freezes the vast majority of the parameters and then merely 
trains the parameters in the final head or classifier layer.

## Models available
### Vision Transformer
A vision transformer model can be specified by giving a model_type
* vit_b_16
* vit_l_16

These are based on the 'basic' and 'large' torchvision pretrained ViT models. The pretrained weights are 
kept as they are, only the final 'heads' layer is trained.

### Efficient Net
An efficientnet can be specified by giving a model type
* effnet_b2
* effnet_b7
* effnet_b2_train_entire_model
* effnet_b7_train_entire_model

These are efficient nets of two different sizes for B2 and B7. They also use the pretrained model weights
from torchvision.

The difference between e.g. effnet_b7 and effnet_b7_train_entire_model is not that the entire model is trainable (only the final 'classifier' head is), 
but it is the following:
* In effnet_b2, the 'features' part of the model (everything apart from the final classifier head) is kept in .eval() mode throughout training. This means that not only are the parameters frozen (as they should be for a transfer learning problem), but also the BatchNorm layers remain unmodified.
  * The advantage is that one only needs to save the state dict for the final classifier layer, so it is very small (<1MB).
  * The disadvantage seems to be that this doesn't train quite as well, though maybe one just needs to use different hyperparameters.
* In effnet_b2_train_entire_model the parameters for the features are frozen, but the entire model is switched between .train() and .eval() between training and cross validation steps.
  * This means the BatchNorm layers change during training (despite the parameters being frozen, something which very much confused me for a while).
  * It also means that we need to save the state_dict for the entire model, not just the classifier head. This can run into the tens or hundreds of MB.
  * It may be possible to only save the state dict for the BatchNorm layers, as the others should be unaffected. Something to investigate in the future.

### Best results
As of 2024-01-08, the best results I managed to obtain were with effnet_b7_train_entire_model.

## Aircraft types

One can specify different aircraft types to do the training on. My goal was to see whether I could
obtain a computer vision model to distinguish between civilian aircraft. Most work otherwise focussed on military aircraft
or mixed civilian aircraft with military aircraft, but distinguishing an F16 from an F35 from a B747 doesn't seem that hard 
(at least harder than an A320 from a B737).

I have pre-defined 3 different sets of aircraft to do classification on:
* TEST: contains only 3 aircraft, good for testing the code.
* CIVILIAN_JETS: The set of aircraft from the FGVCA aircraft dataset which are civilian jet airliners. Contains all the familiar aircraft, apart from the Airbus A350 (the set is from 2012, before the A350 entered commercial service)
* ALL_AIRCRAFT: All aircraft from the FGVCA aircraft dataset, including civilian, military and GA aircraft.

## Classes
### classifiers (in classifiers.py)

#### AircraftClassifier
Main class for doing the aircraft classification. Parameters are
* model_type: type of model to use
* aircraft_subset_name: name of the aircraft subset on which to do classification, in my case CIVILIAN_JETS.
* load_classifier_pretrained_weights: whether to load a fully pretrained classifier or whether to load an uninitialised classifier. In all cases, the weights from all but the final classification layer are the pretrained weights from torchvision.
* classifier_pretrained_weights_file: If we want to load weights, this is the file where they are stored.

##### Attributes
* train_transform_with_crop: transform used for training. It crops the final 20 pixels, which contain the authorship information for the pictures, see https://arxiv.org/pdf/1306.5151.pdf.
* predict_transform_with_crop: transform for predicting when predicting on data from the FGVA dataset. Like in training, it crops the 20 last rows of pixels.
* predict_transform_without_crop: transform for predicting on pictures not in the FGVA dataset, which don't need to be cropped.
* trainable_parts: Which part of the model should be set to .train() during the training step.

##### Methods
* state_dict_extractor(): A function that gets the state dict of the part of the model that we want to save (we don't always want to save the entire state dict, since much of the model is frozen).
* save_model(): Save the part of the state dict that has changed with training to a file.
* predict(): Run a prediction on a new image.

### data_setup

#### AircraftData
Child class of torchvision.datasets.FGVCAircraft. The main difference from the parent is that this stores the aircraft 
type (y value, or target) for all elements in the dataset in one list. 
Without this, it can be very slow to get all the targets, since iterating through the dataset takes time.

Using get_aircraft_data_subset it is possible to get a subset of the aircraft data
(e.g. only civilian jets). 
This function produces a new AircraftData set and does the following
* Only returns data for selected aircraft.
* It maps the target values to the range 0..N-1, where N is the number of aircraft in the subset. Without this step, the target values would contain N sparse values within the range 0..N_tot where N_tot is the total number of aircraft in the FGVA dataset (around 69)


# aircraft_classification
> Code for training and running a classifier for aircraft.
> Based on FGVCA aircraft data, though this code allows one to use a subset

## Installation
Not needed

## Usage example
```python
import fit_model

results = fit_model.fit_model('vit_b_16',
                              'CIVILIAN_JETS',
                              2,
                              'runs',
                              )
```

This will save a model which can be loaded like this:
```python
import classifiers

vit = classifiers.AircraftClassifier('vit_b_16',
                                     'CIVILIAN_JETS',
                                     True,
                                     <path_to_pytorch_file>)

vit.predict(pil_image)
```


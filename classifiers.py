"""
Defines the different aircraft classifiers.
"""

import logging
from collections import OrderedDict
from pathlib import Path
from timeit import default_timer as timer
from typing import Optional, Tuple, Any

import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision.transforms import v2 as transf_v2

import aircraft_types as act
import parameters


def inverse_of_normalisation_transform(transform):
    """
    Perform inverse of the transf_v2.Normalise transform.

    Useful when wanting to visualise images from the dataset or from the dataloader.
    :param transform:
    :return: Inverse of the normalisation transform.
    """
    if hasattr(transform, "mean"):
        means = transform.mean
        stds = transform.std
        # Create the inverse to std.
        inv_std = transf_v2.Normalize(
            mean=[
                0.0,
                0.0,
                0.0,
            ],
            std=[1.0 / s for s in stds],
        )
        # Create the inverse to de-mean.
        inv_means = transf_v2.Normalize(mean=[-m for m in means], std=[1.0, 1.0, 1.0])
        # First apply inverse to divide by std, then inverse to de-mean,
        # reverse of normalisation operation.
        return transf_v2.Compose([inv_std, inv_means])
    else:
        return transf_v2.Identity()


class TrivialClassifier(nn.Module):
    """
    A trivial classifier for testing purposes. It will be hopeless as a classifier.
    """

    image_size = 8
    n_colour_channels = 3
    # Minimum transforms to be able to process some images.
    transforms = transf_v2.Compose(
        [
            transf_v2.Resize((image_size, image_size), antialias=True),
            transf_v2.ToImage(),
            transf_v2.ToDtype(torch.float32, scale=True),
        ]
    )

    def __init__(self, num_classes):
        super().__init__()
        # After flattening, input will have size self.image_size**2.
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.image_size**2 * self.n_colour_channels, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


class AircraftClassifier:
    """
    Contains a classifier for aircraft. It contains a model (nn.Module) and the required transform
    """

    def __init__(
        self,
        model_type: str,
        aircraft_subset_name: str,
        load_classifier_pretrained_weights: bool,
        classifier_pretrained_weights_file: Optional[str | Path] = None,
    ):
        """
        Initialise an AircraftClassifier.

        :param model_type: "vit_b_16", "vit_l_16", "effnet_b2", "effnet_b7"
        :param aircraft_subset_name: name of the aircraft subset ('TEST', 'CIVILIAN_JETS',..).
            Must be one defined in aircraft_types.AIRCRAFT_SUBSETS
        :param load_classifier_pretrained_weights: whether to load classifier
        :param classifier_pretrained_weights_file: file for classifier data
        """
        self.aircraft_subset_name = aircraft_subset_name.upper()
        # Check that aircraft_subset_name is actually defined.
        assert self.aircraft_subset_name in act.AIRCRAFT_SUBSETS, (
            f"aircraft_subset_name={aircraft_subset_name} undefined, "
            f"not one of {list(act.AIRCRAFT_SUBSETS.keys())}"
        )
        self.class_names = act.AIRCRAFT_SUBSETS[self.aircraft_subset_name]
        self.classifier_pretrained_weights_file = classifier_pretrained_weights_file
        self.load_classifier_pretrained_weights = load_classifier_pretrained_weights
        self.num_classes = len(self.class_names)
        self.model_type = (
            model_type.lower()
        )  # drop complications from upper/lower cases

        # check that classifier_pretrained_weights_file exists.
        if load_classifier_pretrained_weights:
            assert (
                classifier_pretrained_weights_file is not None
            ), "Want to load pretrained weights, but no classifier_pretrained_weights_file given"
            assert Path(
                classifier_pretrained_weights_file
            ).is_file(), f"classifier_pretrained_weights_file = {classifier_pretrained_weights_file} doesn't exist."

        # Initialise model
        self.model = None
        # This contains the transform that is needed for the underlying model, preset by ViT or Effnet.
        self.transforms = None
        # Specify which part of the model should be set to .eval() during the train step.
        self.trainable_parts = "all"

        # Obtain model and set weights.
        self._get_model_and_transform()

        # If we train on the FGVCAircraft dataset, we need to implement cropping, same for prediction.
        # If we do prediction on a new picture, crop shouldn't be done.
        # Only the training gets the data augmentation transforms.
        self.train_transform_with_crop = transf_v2.Compose(
            [
                parameters.TO_TENSOR_TRANSFORMS,
                parameters.CropAuthorshipInformation(),
                parameters.DATA_AUGMENTATION_TRANSFORMS,
                self.transforms,
            ]
        )
        self.predict_transform_with_crop = transf_v2.Compose(
            [
                parameters.TO_TENSOR_TRANSFORMS,
                parameters.CropAuthorshipInformation(),
                self.transforms,
            ]
        )
        self.predict_transform_without_crop = transf_v2.Compose(
            [parameters.TO_TENSOR_TRANSFORMS, self.transforms]
        )
        self.inv_of_normalisation = inverse_of_normalisation_transform(self.transforms)

    @staticmethod
    def _get_transforms_from_pretrained_weights(weights):
        """Extract transforms and set antialias=True"""
        # Need to extract the transforms from the weights.
        transforms = weights.transforms()

        # We need to set antialias = True. The way the transforms seem to be set up
        # is that the model has been trained on PIL images, where antialias is always true.
        # Here I need to first convert to tensor, in order to cut off the authorship
        # information. But transforms.antialias is set to "warn" which appears to
        # switch off antialias (and produces an error). Without antialias the pictures also look
        # very distorted.
        transforms.antialias = True
        return transforms

    def _model_and_transform_factory(self) -> Tuple[nn.Module, Any]:
        """
        Get the torchvision model and the required transform for the images.

        The transforms need to be the same as those used when
        training the original model, otherwise transfer learning and
        prediction will not work.

        :return: model, transform
        """
        if self.model_type == "trivial":
            # Get case for trivial model. Easiest.
            model = TrivialClassifier(self.num_classes)
            transforms = model.transforms
        else:
            if self.model_type == "vit_b_16":
                weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
                model = torchvision.models.vit_b_16(weights=weights)
            elif self.model_type == "vit_l_16":
                weights = torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
                model = torchvision.models.vit_l_16(weights=weights)
            elif self.model_type.startswith("effnet_b2"):
                weights = torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1
                # For some reason, with effnet need to get state dict, otherwise things crash.
                _ = weights.get_state_dict()
                model = torchvision.models.efficientnet_b2(weights=weights)
            elif self.model_type.startswith("effnet_b7"):
                weights = torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1
                # For some reason, with effnet need to get state dict, otherwise things crash.
                _ = weights.get_state_dict()
                model = torchvision.models.efficientnet_b7(weights=weights)

            else:
                raise NotImplementedError(
                    f"model_type={self.model_type} not implemented."
                )

            # Freeze all the parameters.
            # The classifier gets replaced later with unfrozen parameters.
            for param in model.parameters():
                param.requires_grad = False

            transforms = self._get_transforms_from_pretrained_weights(weights)

        return model, transforms

    def _get_model_and_transform(
        self,
    ) -> None:
        """
        Load a torchvision model with pre-trained weights, replacing classifier head.

        Replace classifier head (this may have different names) with a
        custom classifier.
        Always load model to CPU, as not all devices will have a GPU.
        We can always move to GPU later.

        :param model_type:
        :return: None
        """
        model, transforms = self._model_and_transform_factory()

        # Replace classifier
        if self.model_type.startswith("vit"):
            # The classifier head for a Vision Transformer is called "heads"
            # and consists of a Sequential with one linear layer.
            old_head = model.heads.head
            in_features = old_head.in_features
            bias = old_head.bias is not None
            # Need to construct an OrderedDict to get the naming right.
            new_head_dict = OrderedDict()
            new_head_dict["head"] = nn.Linear(in_features, self.num_classes, bias)
            model.heads = nn.Sequential(new_head_dict)

            if self.load_classifier_pretrained_weights:
                # Load weights from file (we know it exists).
                model.heads.load_state_dict(
                    torch.load(
                        self.classifier_pretrained_weights_file,
                        map_location=torch.device("cpu"),
                    ),
                )

        elif self.model_type.startswith("effnet"):
            # Classfier head is called "classifier" and consists of a dropout and linear layer.
            # These are constructed from Sequential without naming of the layers, so from a list.
            old_head = model.classifier
            dropout_layer = old_head[0]
            linear_layer = old_head[1]
            new_head = nn.Sequential(
                nn.Dropout(p=dropout_layer.p, inplace=dropout_layer.inplace),
                nn.Linear(
                    in_features=linear_layer.in_features,
                    out_features=self.num_classes,
                    bias=linear_layer.bias is not None,
                ),
            )
            model.classifier = new_head

            if self.load_classifier_pretrained_weights:
                # Load weights from file (we know the file exists).
                if self.model_type.endswith("train_entire_model"):
                    # Here we want to apply the state dict to the entire model, not just the classifier
                    model.load_state_dict(
                        torch.load(
                            self.classifier_pretrained_weights_file,
                            map_location=torch.device("cpu"),
                        )
                    )
                else:
                    # Here we only apply the state dict to the classifier, leave all the rest as is
                    model.classifier.load_state_dict(
                        torch.load(
                            self.classifier_pretrained_weights_file,
                            map_location=torch.device("cpu"),
                        )
                    )

            if self.model_type.endswith("train_entire_model"):
                # Don't do anything. By default already train all parts of the model
                pass
            else:
                # Only the classifier part should be trainable. Because effnet uses BatchNorm,
                # even if the parameters are frozen, if the entire model is trainable,
                # the output from BatchNorm will change, and thus also the features change.
                # I don't want the features to change with training, otherwise I would have
                # to save the entire state_dict and load that
                # (and I'm also quite happy with all the pretrained parameters for the features)
                self.trainable_parts = ["classifier"]

        elif self.model_type == "trivial":
            # check whether we want to load the state dict.
            if self.load_classifier_pretrained_weights:
                model.load_state_dict(
                    torch.load(
                        self.classifier_pretrained_weights_file,
                        map_location=torch.device("cpu"),
                    )
                )
        else:
            raise NotImplementedError(f"model_type={self.model_type} not implemented.")

        if self.load_classifier_pretrained_weights:
            logging.info(
                f"Loaded model weights from {self.classifier_pretrained_weights_file}"
            )

        # Set the entire model to eval mode. That way, later, some parts can be set
        # to train mode and some to eval.
        model.eval()

        self.model = model
        self.transforms = transforms

    def state_dict_extractor(self, model) -> dict[str, Any]:
        """
        Extract the part of the state dict we want to save, the part that was trained.

        This function can also be passed to ml_utils.ClassificationTrainer,
        so the correct state dict gets saved.
        :parameter: model. This is not self.model, so it can be passed as a parameter,
            as required by ml_utils.ClassificationTrainer
        """
        if self.model_type.startswith("vit"):
            # Extract "heads", as that's the part we trained
            state_dict = model.heads.state_dict()
        elif self.model_type.startswith("effnet") and self.model_type.endswith(
            "train_entire_model"
        ):
            # We want to train the entire model, including the BatchNorm, so get the entire state_dict
            state_dict = model.state_dict()
        elif self.model_type.startswith("effnet") and not (
            self.model_type.endswith("train_entire_model")
        ):
            # Extract "classifier", as that's the part we trained
            state_dict = model.classifier.state_dict()
        elif self.model_type == "trivial":
            # Get the entire model.
            state_dict = model.state_dict()
        else:
            raise NotImplementedError(
                f"model_type={self.model_type} not implemented for saving."
            )
        return state_dict

    def save_model(self, output_file: str | Path) -> None:
        """
        Save model to the file.

        Depending on what type of model it is, save either the entire model
        or just the classifier head.
        :param output_file:
        :return: None
        """
        output_file = Path(output_file)
        # Get the part of the state_dict that we want to save.
        state_dict = self.state_dict_extractor(self.model)
        # Then save it.
        torch.save(state_dict, output_file)
        logging.info(f"Saved model to file {output_file}")

    def predict(
        self, img: Image, apply_crop: bool = False
    ) -> Tuple[dict[str, float], float]:
        """
        For a given image, make the prediction.
        :param img:
        :param apply_crop: whether to crop the bottom 20 pixels. Only needed for FGVCAircraft dataset
        :return: dict of prediction probabilities for all classes and time to make prediction
        """
        # Apply transforms
        if apply_crop:
            trans_img = self.predict_transform_with_crop(img)
        else:
            trans_img = self.predict_transform_without_crop(img)

        # Set model to eval mode.
        self.model.eval()

        start_time = timer()

        # Perform inference.
        with torch.inference_mode():
            # Need to add a batch dimension for inference and then remove it.
            logits = self.model(trans_img.unsqueeze(0)).squeeze()

            # To get pred probs take softmax over last dimension, which contains
            # the logits for each class.
            pred_probs = torch.softmax(logits, dim=-1)

            pred_labels_and_probs = {
                self.class_names[i]: pred_probs[i].item()
                for i in range(self.num_classes)
            }

        end_time = timer()

        # Get prediction time.
        pred_time = end_time - start_time

        return pred_labels_and_probs, pred_time

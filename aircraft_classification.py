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


class TrivialClassifier(nn.Module):
    """
    A trivial classifier for testing purposes. It will be hopeless as a classifier.
    """

    image_size = 8
    n_colour_channels = 3
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
        class_names: list[str],
        load_classifier_pretrained_weights: bool,
        classifier_pretrained_weights_file: Optional[str | Path] = None,
    ):
        """

        :param model_type: "vit_b_16", "vit_l_16", "effnet_b2", "effnet_b7"
        :param class_names:
        :param load_classifier_pretrained_weights: whether to load classifier
        :param classifier_pretrained_weights_file: file for classifier data
        """
        self.class_names = class_names
        self.classifier_pretrained_weights_file = classifier_pretrained_weights_file
        self.load_classifier_pretrained_weights = load_classifier_pretrained_weights
        self.num_classes = len(class_names)
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

        # Initialise model and transform.
        self.model = None
        self.transforms = None

        # Obtain model and set weights.
        self._get_model_and_transform()

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
        if self.model_type == "trivial":
            # Get case for trivial model. Easiest.
            model = TrivialClassifier(self.num_classes)
            transforms = model.transforms
        else:
            if self.model_type == "vit_b_16":
                weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
                model = torchvision.models.vit_b_16(weights=weights)
            elif self.model_type == "vit_b_32":
                weights = torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
                model = torchvision.models.vit_l_16(weights=weights)
            elif self.model_type == "effnet_b2":
                weights = torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1
                model = torchvision.models.efficientnet_b2(weights=weights)
            elif self.model_type == "effnet_b7":
                weights = torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1
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
        Load a torchvision model with pre-trained weights.

        Replace classifier head (this may have different names) with a custom classifier.

        :param model_type:
        :return: None
        """
        model, transforms = self._model_and_transform_factory()

        # Replace classifier
        if self.model_type.startswith("vit"):
            # The classifier head is called "heads" and consists of a Sequential
            # with one linear layer.
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
                    torch.load(self.classifier_pretrained_weights_file)
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
                # Load weights from file (we know it exists).
                model.classifier.load_state_dict(
                    torch.load(self.classifier_pretrained_weights_file)
                )
        elif self.model_type == "trivial":
            # check whether we want to load the state dict.
            if self.load_classifier_pretrained_weights:
                model.load_state_dict(
                    torch.load(self.classifier_pretrained_weights_file)
                )
        else:
            raise NotImplementedError(f"model_type={self.model_type} not implemented.")

        if self.load_classifier_pretrained_weights:
            logging.info(
                f"Loaded model weights from {self.classifier_pretrained_weights_file}"
            )

        self.model = model
        self.transforms = transforms

    def save_model(self, output_file: str | Path) -> None:
        """
        Save model to the file.

        Depending on what type of model it is, save either the entire model
        or just the classifier head.
        :param output_file:
        :return: None
        """
        output_file = Path(output_file)
        if self.model_type.startswith("vit"):
            # Save "heads"
            torch.save(self.model.heads.state_dict(), output_file)
        elif self.model_type.startswith("effnet"):
            # Save "classifier"
            torch.save(self.model.classifier.state_dict(), output_file)
        elif self.model_type == "trivial":
            # Save the entire model.
            torch.save(self.model.state_dict(), output_file)
        else:
            raise NotImplementedError(
                f"model_type={self.model_type} not implemented for saving."
            )
        logging.info(f"Saved model to file {output_file}")

    def predict(self, img: Image) -> Tuple[dict[str, float], float]:
        """
        For a given image, make the prediction.
        :param img:
        :return: dict of prediction probabilities for all classes and time to make prediction
        """
        # Apply transforms
        trans_img = self.transforms(img)

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

        return pred_labels_and_probs, end_time - start_time

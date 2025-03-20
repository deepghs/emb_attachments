import logging
import os.path
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from PIL import Image
from imgutils.data import load_image
from realutils.tagging.idolsankaku import _prepare_image_for_tagging
from timm import create_model
from timm.data import resolve_model_data_config, create_transform
from torch import nn
from torchvision import transforms


@dataclass
class EncoderModel:
    model_name: str
    preprocessor: Callable[[Image.Image], torch.Tensor]
    model: nn.Module
    width: int


class ModuleWrapper(nn.Module):
    def __init__(self, base_module: nn.Module, classifier: nn.Module):
        super().__init__()
        self.base_module = base_module
        self.classifier = classifier

        self._output_features = None
        self._register_hook()

    def _register_hook(self):
        def hook_fn(module, input_tensor, output_tensor):
            assert isinstance(input_tensor, tuple) and len(input_tensor) == 1
            input_tensor = input_tensor[0]
            self._output_features = input_tensor

        self.classifier.register_forward_hook(hook_fn)

    def forward(self, x: torch.Tensor):
        _ = self.base_module(x)

        if self._output_features is None:
            raise RuntimeError("Target module did not receive any input during forward pass")
        features, self._output_features = self._output_features, None
        assert all([x == 1 for x in features.shape[2:]]), f'Invalid feature shape: {features.shape!r}'
        features = torch.flatten(features, start_dim=1)

        return features


def create_square_pad_transform(target_size: int):
    return transforms.Compose([
        transforms.Lambda(lambda img: _make_square(img)),
        transforms.Resize((target_size, target_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flip(x, [0])),
        transforms.Lambda(lambda x: x * 255.0 / 127.5 - 1.0),
    ])


def _make_square(img):
    max_dim = max(img.size)
    pad_left = (max_dim - img.size[0]) // 2
    pad_top = (max_dim - img.size[1]) // 2

    padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    try:
        padded_image.paste(img, (pad_left, pad_top), mask=img)
    except ValueError:
        padded_image.paste(img, (pad_left, pad_top))

    return padded_image


def load_from_timm_repo(repo_id: str, is_smilingwolf: bool = False, is_main_process: bool = True):
    if is_main_process:
        logging.info(f'Loading TIMM model from {repo_id!r} ...')
    model = create_model(f'hf-hub:{repo_id}', pretrained=True)

    if is_main_process:
        logging.info('Loading default preprocessor ...')
    data_config = resolve_model_data_config(model)
    t = create_transform(**data_config, is_training=False)

    classifier = model.get_classifier()
    classifier_position = None
    for name, module in model.named_modules():
        if module is classifier:
            classifier_position = name
            break
    if not classifier_position:
        raise RuntimeError(f'No classifier module found in model {type(model)}.')
    if is_main_process:
        logging.info(f'Classifier module found at {classifier_position!r}:\n{classifier}')

    if is_main_process:
        logging.info('Wrapped model created.')
    wrapped_model = ModuleWrapper(model, classifier=classifier)

    image = load_image(os.path.join(os.path.dirname(__file__), 'sample.jpg'), mode='RGB', force_background='white')
    dummy_input = t(image).unsqueeze(0)
    if is_main_process:
        logging.info(f'Expected input size: {dummy_input.shape!r}')

    if is_smilingwolf:
        if is_main_process:
            logging.info('Extra-processing preprocessor for smilingwolf-like models ...')
        expected_size = dummy_input.shape[2]
        t = create_square_pad_transform(expected_size)
        dummy_input = t(image).unsqueeze(0)

        if is_main_process:
            logging.info('Calculating expected input for smilingwolf taggers ...')
        expected_input = torch.from_numpy(_prepare_image_for_tagging(image, target_size=expected_size)).type(
            torch.float32)

        if is_main_process:
            logging.info('Checking input alignments ...')
        np.testing.assert_allclose(
            dummy_input.numpy(),
            expected_input.numpy()
        )

    if is_main_process:
        logging.info('Try infer with this encoder model ...')
    with torch.no_grad():
        dummy_output = wrapped_model(dummy_input)
    if is_main_process:
        logging.info(f'Output size: {dummy_output.shape!r}')

    return EncoderModel(
        model_name=f'{"timm" if not is_smilingwolf else "wdtagger"}:{repo_id}',
        preprocessor=t,
        model=wrapped_model,
        width=dummy_output.shape[-1],
    )


def load_encoder(model_name: str, is_main_process: bool = True):
    type_, name = model_name.split(':', maxsplit=1)
    if type_ == 'timm':
        return load_from_timm_repo(repo_id=name, is_smilingwolf=False, is_main_process=is_main_process)
    elif type_ == 'wdtagger':
        return load_from_timm_repo(repo_id=name, is_smilingwolf=True, is_main_process=is_main_process)
    else:
        raise ValueError(f'Unknown type for encoder model - {type_!r}')


if __name__ == '__main__':
    print(load_encoder('wdtagger:SmilingWolf/wd-swinv2-tagger-v3'))

import os

import onnx
import onnxoptimizer
import onnxsim
import torch
from ditk import logging
from hbutils.system import TemporaryDirectory

from eattach.model import Backbone, BackboneWithHead
from .encode import EncoderModel
from .problem import Problem


def onnx_optimize(model):
    model = onnxoptimizer.optimize(model)
    model, check = onnxsim.simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    return model


def export_backbone_to_onnx(backbone_file: str, encoder: EncoderModel, problem: Problem,
                            onnx_filename: str, verbose: bool = True, opset_version: int = 14,
                            no_optimize: bool = False):
    backbone, metadata = Backbone.load(backbone_file, with_metadata=True)
    model = BackboneWithHead(
        backbone=backbone.module,
        head=problem.get_head(keep_logits=True),
    )
    model = model.float()

    dummy_input = torch.randn(1, encoder.width)
    dummy_input = dummy_input.float()

    with torch.no_grad(), TemporaryDirectory() as td:
        onnx_model_file = os.path.join(td, 'model.onnx')
        logging.info(f'Onnx data exporting to {onnx_model_file!r} ...')
        torch.onnx.export(
            model,
            (dummy_input,),
            onnx_model_file,
            verbose=verbose,
            input_names=["input"],
            output_names=["prediction", "logits"],

            opset_version=opset_version,
            dynamic_axes={
                "input": {0: "batch"},
                "prediction": {0: "batch"},
                "logits": {0: "batch"},
            },
        )

        model = onnx.load(onnx_model_file)
        if not no_optimize:
            logging.info('Optimizing onnx model ...')
            model = onnx_optimize(model)

        output_model_dir, _ = os.path.split(onnx_filename)
        if output_model_dir:
            os.makedirs(output_model_dir, exist_ok=True)

        for k, v in metadata.items():
            logging.info(f'Adding metadata {k!r} = {v!r} ...')
            assert isinstance(v, str)
            meta = model.metadata_props.add()
            meta.key, meta.value = k, v

        logging.info(f'Complete model saving to {onnx_filename!r} ...')
        onnx.save(model, onnx_filename)

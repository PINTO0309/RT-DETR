"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn

from src.core import YAMLConfig


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            # outputs = self.postprocessor(outputs, orig_target_sizes)
            outputs = self.postprocessor(outputs)
            return outputs

    model = Model()

    h, w = args.size
    data = torch.rand(1, 3, h, w)
    size = torch.tensor([[h, w]])
    _ = model(data, size)

    dynamic_axes = {
        'images': {0: 'N', },
        # 'orig_target_sizes': {0: 'N'},
        'label_xyxy_score': {0: 'N', },
    }

    output_file = f'{os.path.splitext(os.path.basename(args.config))[0]}_{args.query}query.onnx'

    torch.onnx.export(
        model,
        (data, size),
        output_file,
        # input_names=['images', 'orig_target_sizes'],
        input_names=['images'],
        # output_names=['labels', 'boxes', 'scores'],
        output_names=['label_xyxy_score'],
        dynamic_axes=dynamic_axes if args.dynamic_batch else None,
        opset_version=16,
    )
    torch.onnx.export(
        model,
        (data, size),
        f'{os.path.splitext(os.path.basename(output_file))[0]}_n_batch.onnx',
        # input_names=['images', 'orig_target_sizes'],
        input_names=['images'],
        # output_names=['labels', 'boxes', 'scores'],
        output_names=['label_xyxy_score'],
        dynamic_axes=dynamic_axes,
        opset_version=16,
    )

    if args.check:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')

    if args.simplify:
        import onnx
        import onnxsim
        dynamic: bool = args.dynamic_batch
        # input_shapes = {'images': [1, 3, 640, 640], 'orig_target_sizes': [1, 2]} if dynamic else None
        input_shapes = {'images': data.shape, 'orig_target_sizes': size.shape} if dynamic else None

        onnx_model_simplify, check = onnxsim.simplify(output_file, input_shapes=input_shapes, dynamic_input_shape=dynamic)
        onnx.save(onnx_model_simplify, output_file)
        print(f'Simplify onnx model {check}...')

        onnx_model_simplify, check = onnxsim.simplify(f'{os.path.splitext(os.path.basename(output_file))[0]}_n_batch.onnx')
        onnx.save(onnx_model_simplify, f'{os.path.splitext(os.path.basename(output_file))[0]}_n_batch.onnx')
        print(f'Simplify onnx model {check}...')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--size', '-s', nargs="*", default=[640,640], type=int, )
    parser.add_argument('--check',  action='store_true', default=False,)
    parser.add_argument('--simplify',  action='store_true', default=False,)
    parser.add_argument('--dynamic_batch',  action='store_true', default=False,)
    parser.add_argument('--query', '-q', type=int, default=300)

    args = parser.parse_args()

    main(args)

"""
Demo single image
Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
If you use this code, please cite the following paper:
Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""
__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import argparse
import logging
import os
import torch
from PIL import Image
from src.arch import deep_wb_model, deep_wb_single_task
import src.utilities.utils as utls
from src.utilities.deepWB import deep_wb
import src.arch.splitNetworks as splitter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_args():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(description='Change the white balance of an input image.')
    parser.add_argument('--model_dir', '-m', default='./models',
                        help="Directory of the trained model.", dest='model_dir')
    parser.add_argument('--input', '-i', required=True,
                        help='Input image filename', dest='input')
    parser.add_argument('--output_dir', '-o', default='./result_images',
                        help='Directory to save the output images', dest='out_dir')
    parser.add_argument('--task', '-t', default='all',
                        help="Task to perform: 'AWB', 'editing', or 'all'.", dest='task')
    parser.add_argument('--target_color_temp', '-tct', type=int,
                        help="Target color temperature [2850 - 7500]. Requires task 'editing'.", dest='target_color_temp')
    parser.add_argument('--max_size', '-S', default=656, type=int,
                        help="Max dimension of input image to the network.", dest='max_size')
    parser.add_argument('--show', '-v', action='store_true',
                        help="Visualize the input and output images", dest='show')
    parser.add_argument('--save', '-s', action='store_true', default=True,
                        help="Save the output images", dest='save')
    parser.add_argument('--device', '-d', default='cuda',
                        help="Device to use: 'cuda' or 'cpu'.", dest='device')
    return parser.parse_args()

def load_model(model_path, model_class, device):
    """Load a model from the specified path and move it to the specified device."""
    model = model_class()
    logging.info(f"Loading model {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def ensure_dir(directory):
    """Ensure that the specified directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    args = get_args()
    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if args.target_color_temp:
        if not 2850 <= args.target_color_temp <= 7500:
            raise ValueError('Color temperature should be in the range [2850 - 7500].')
        if args.task.lower() != 'editing':
            raise ValueError('The task should be editing when a target color temperature is specified.')

    if args.save:
        ensure_dir(args.out_dir)

    # Load models based on the task
    if args.task.lower() == 'all':
        model_paths = {
            'awb': os.path.join(args.model_dir, 'net_awb.pth'),
            't': os.path.join(args.model_dir, 'net_t.pth'),
            's': os.path.join(args.model_dir, 'net_s.pth')
        }
        if all(os.path.exists(path) for path in model_paths.values()):
            net_awb = load_model(model_paths['awb'], deep_wb_single_task.deepWBnet, device)
            net_t = load_model(model_paths['t'], deep_wb_single_task.deepWBnet, device)
            net_s = load_model(model_paths['s'], deep_wb_single_task.deepWBnet, device)
        elif os.path.exists(os.path.join(args.model_dir, 'net.pth')):
            net = load_model(os.path.join(args.model_dir, 'net.pth'), deep_wb_model.deepWBNet, device)
            net_awb, net_t, net_s = splitter.splitNetworks(net)
        else:
            raise FileNotFoundError('Model not found!')
    elif args.task.lower() == 'editing':
        model_paths = {
            't': os.path.join(args.model_dir, 'net_t.pth'),
            's': os.path.join(args.model_dir, 'net_s.pth')
        }
        if all(os.path.exists(path) for path in model_paths.values()):
            net_t = load_model(model_paths['t'], deep_wb_single_task.deepWBnet, device)
            net_s = load_model(model_paths['s'], deep_wb_single_task.deepWBnet, device)
        elif os.path.exists(os.path.join(args.model_dir, 'net.pth')):
            net = load_model(os.path.join(args.model_dir, 'net.pth'), deep_wb_model.deepWBNet, device)
            _, net_t, net_s = splitter.splitNetworks(net)
        else:
            raise FileNotFoundError('Model not found!')
    elif args.task.lower() == 'awb':
        model_path = os.path.join(args.model_dir, 'net_awb.pth')
        if os.path.exists(model_path):
            net_awb = load_model(model_path, deep_wb_single_task.deepWBnet, device)
        elif os.path.exists(os.path.join(args.model_dir, 'net.pth')):
            net = load_model(os.path.join(args.model_dir, 'net.pth'), deep_wb_model.deepWBNet, device)
            net_awb, _, _ = splitter.splitNetworks(net)
        else:
            raise FileNotFoundError('Model not found!')
    else:
        raise ValueError("Invalid task! Task should be: 'AWB', 'editing', or 'all'")

    # Process the image
    img = Image.open(args.input)
    base_name = os.path.splitext(os.path.basename(args.input))[0]

    if args.task.lower() == 'all':
        out_awb, out_t, out_s = deep_wb(img, task=args.task.lower(), net_awb=net_awb, net_s=net_s, net_t=net_t,
                                        device=device, s=args.max_size)
        out_f, out_d, out_c = utls.colorTempInterpolate(out_t, out_s)
        if args.save:
            utls.to_image(out_awb).save(os.path.join(args.out_dir, f'{base_name}_AWB.png'))
            utls.to_image(out_t).save(os.path.join(args.out_dir, f'{base_name}_T.png'))
            utls.to_image(out_s).save(os.path.join(args.out_dir, f'{base_name}_S.png'))
            utls.to_image(out_f).save(os.path.join(args.out_dir, f'{base_name}_F.png'))
            utls.to_image(out_d).save(os.path.join(args.out_dir, f'{base_name}_D.png'))
            utls.to_image(out_c).save(os.path.join(args.out_dir, f'{base_name}_C.png'))
        if args.show:
            utls.imshow(img, utls.to_image(out_awb), utls.to_image(out_t), utls.to_image(out_f),
                        utls.to_image(out_d), utls.to_image(out_c), utls.to_image(out_s))
    elif args.task.lower() == 'awb':
        out_awb = deep_wb(img, task=args.task.lower(), net_awb=net_awb, device=device, s=args.max_size)
        if args.save:
            utls.to_image(out_awb).save(os.path.join(args.out_dir, f'{base_name}_AWB.png'))
        if args.show:
            utls.imshow(img, utls.to_image(out_awb))
    else:  # editing
        out_t, out_s = deep_wb(img, task=args.task.lower(), net_s=net_s, net_t=net_t, device=device, s=args.max_size)
        if args.target_color_temp:
            out = utls.colorTempInterpolate_w_target(out_t, out_s, args.target_color_temp)
            if args.save:
                utls.to_image(out).save(os.path.join(args.out_dir, f'{base_name}_{args.target_color_temp}.png'))
            if args.show:
                utls.imshow(img, utls.to_image(out), colortemp=args.target_color_temp)
        else:
            out_f, out_d, out_c = utls.colorTempInterpolate(out_t, out_s)
            if args.save:
                utls.to_image(out_t).save(os.path.join(args.out_dir, f'{base_name}_T.png'))
                utls.to_image(out_s).save(os.path.join(args.out_dir, f'{base_name}_S.png'))
                utls.to_image(out_f).save(os.path.join(args.out_dir, f'{base_name}_F.png'))
                utls.to_image(out_d).save(os.path.join(args.out_dir, f'{base_name}_D.png'))
                utls.to_image(out_c).save(os.path.join(args.out_dir, f'{base_name}_C.png'))
            if args.show:
                utls.imshow(img, utls.to_image(out_t), utls.to_image(out_f), utls.to_image(out_d),
                            utls.to_image(out_c), utls.to_image(out_s))

if __name__ == "__main__":
    main()
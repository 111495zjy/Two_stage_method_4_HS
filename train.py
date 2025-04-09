from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
os.environ['SDL_AUDIODRIVER'] = 'dummy'
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
import random
from event_data import EventData
from model import EvINRModel,EvINRModel_stage2
import cv2
import math
from skimage.metrics import structural_similarity
def config_parser():
    parser = ArgumentParser(description="EvINR")
    parser.add_argument('--exp_name', '-n', type=str, help='Experiment name')
    parser.add_argument('--data_path', '-d', type=str, help='Path of events.npy to train')
    parser.add_argument('--output_dir', '-o', type=str, default='logs', help='Directory to save output')
    parser.add_argument('--t_start', type=float, default=0, help='Start time')
    parser.add_argument('--t_end', type=float, default=2.4, help='End time')
    parser.add_argument('--H', type=int, default=480, help='Height of frames')
    parser.add_argument('--W', type=int, default=640, help='Width of frames')
    parser.add_argument('--color_event', action='store_true', default=False, help='Whether to use color event')
    parser.add_argument('--event_thresh', type=float, default=1, help='Event activation threshold')
    parser.add_argument('--train_resolution', type=int, default=100, help='Number of training frames')
    parser.add_argument('--val_resolution', type=int, default=100, help='Number of validation frames')
    parser.add_argument('--no_c2f', action='store_true', default=True, help='Whether to use coarse-to-fine training')
    parser.add_argument('--iters', type=int, default=1000, help='Training iterations')
    parser.add_argument('--log_interval', type=int, default=2000, help='Logging interval')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--net_layers', type=int, default=3, help='Number of layers in the network')
    parser.add_argument('--net_width', type=int, default=100, help='Hidden dimension of the network')
    parser.add_argument('--net2_width', type=int, default=40, help='Hidden dimension of the network')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')

    return parser





def main(args):
    low_speed_frame_number_start = 10
    low_speed_frame_number = 50
    events = EventData(
        args.data_path, args.t_start, args.t_end, args.H, args.W, args.color_event, args.event_thresh, args.device)
    events.stack_event_frames(args.train_resolution)

    print(f"Number of frames: {len(events.timestamps)}")

    model = EvINRModel(
        args.net_layers, args.net_width, H=events.H, W=events.W, recon_colors=args.color_event
    ).to(args.device)

    print(f'Start training Model1...')
    optimizer = torch.optim.AdamW(params=model.net.parameters(), lr=3e-4)

    writer = SummaryWriter(os.path.join(args.output_dir, args.exp_name))
    print(f'Start training ...')
    for i_iter in trange(1, args.iters + 1):
        optimizer.zero_grad()

        log_intensity_preds = model(events.timestamps[0:low_speed_frame_number])
        loss = model.get_losses(log_intensity_preds, events.event_frames[0:low_speed_frame_number])
        loss.backward()
        optimizer.step()
        if i_iter % args.log_interval == 0:
            tqdm.write(f'iter {i_iter}, loss {loss.item():.4f}')
            writer.add_scalar('loss', loss.item(), i_iter)

        if not args.no_c2f and i_iter == (args.iters // 2):
            events.stack_event_frames(args.train_resolution * 2)

    print(f'End training Model1...')

    #get the groundtruth of stage1
    image_stage1 = model(events.timestamps[low_speed_frame_number_start:low_speed_frame_number]).detach().clone()

    #next stage
    print(f'Prepare TS data and event data...')

    event_TS = events.get_TS(events.timestamps[low_speed_frame_number_start:low_speed_frame_number],low_speed_frame_number-low_speed_frame_number_start) #event_data.py

    #model2 = EvINRModel_stage2(
        #args.net_layers, args.net2_width, H=events.H, W=events.W, recon_colors=args.color_event
    #).to(args.device) #model.py

    print(f'Start training Model2...')
    #optimizer = torch.optim.AdamW(params=model2.net.parameters(), lr=3e-4)

    model2 = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=1,
    activation=None,
    ).to(args.device)
    # ======= 训练设置 =======
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model2.parameters(), lr=1e-4)
    writer = SummaryWriter(os.path.join(args.output_dir, args.exp_name))
    for i_iter in trange(1, args.iters + 1):
        for j in trange(1, 40):
          log_intensity_preds = model2(event_TS[j].unsqueeze(0).unsqueeze(0))
          loss = criterion(log_intensity_preds,image_stage1[j].unsqueeze(0).unsqueeze(0).squeeze(-1))
          #loss = model2.get_losses(log_intensity_preds, events.event_frames[low_speed_frame_number_start:low_speed_frame_number],image_stage1)
          loss.backward()
          optimizer.step()
        if i_iter % args.log_interval == 0:
            tqdm.write(f'iter {i_iter}, loss {loss.item():.4f}')
            writer.add_scalar('loss', loss.item(), i_iter)

        if not args.no_c2f and i_iter == (args.iters // 2):
            events.stack_event_frames(args.train_resolution * 2)
#end stage2

#reference
    with torch.no_grad():
        event_TS_all = events.get_TS(events.timestamps,args.train_resolution) #event_data.py
        #val_timestamps = torch.linspace(0, 1, args.val_resolution).to(args.device).reshape(-1, 1)
        #log_intensity_preds = model2(event_TS_all)
        for i in range(args.train_resolution):
            intensity_preds = model2(event_TS_all[i].unsqueeze(0).unsqueeze(0))
        #intensity_preds = model2.tonemapping(log_intensity_preds).squeeze(-1)

            intensity1 = intensity_preds.squeeze().cpu().detach().numpy()
            image_data = (intensity1*255).astype(np.uint8)

            # 将 NumPy 数组转换为 PIL 图像对象
            image = Image.fromarray(image_data)
            output_path = os.path.join('/content/Two_stage_method_4_HS/logs', 'output_image_{}.png'.format(i))
            image.save(output_path)





if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    main(args)

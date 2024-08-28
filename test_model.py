# -*- coding:utf-8 -*-
import argparse
import numpy as np
import torch
from logger import Logger
import logging
from Net import STEI
import Metrics

def getsample(args):
    with open(args.testdata_path, encoding='utf-8') as f:
        data = np.loadtxt(args.testdata_path, dtype=float, delimiter=',')
    m = data.shape[0]
    n = data.shape[1]
    sensed_num = int(m * args.sensed_ratio)
    mask = np.zeros([m, n])
    for i in range(n):
        mask[0:sensed_num, i] = 1
        np.random.shuffle(mask[:, i])
    sample = {'mask': np.reshape(mask, (m, args.h, args.w)), 'label': np.reshape(data, (m, args.h, args.w))}
    return sample

def test(net, args):
    sample = getsample(args)
    label = torch.tensor(sample['label'])
    mask = torch.tensor(sample['mask'])
    sensed_data = np.multiply(label, mask)
    sensed_data = sensed_data.unsqueeze(1).type(torch.Tensor).to(args.device)
    label = label.unsqueeze(1).type(torch.Tensor).to(args.device)
    mask = mask.unsqueeze(1).type(torch.Tensor).to(args.device)
    output, i = net(sensed_data, mask)
    result = output * (1 - mask) + sensed_data
    RMSE = Metrics.get_RMSE(label, result)
    MAE = Metrics.get_MAE(label, result)
    return RMSE, MAE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for computation (e.g., "cpu", "cuda:0").')
    parser.add_argument('--sensed_ratio', type=int, default=0.5, help='Sensed ratio.')
    parser.add_argument('--h', type=int, default=30, help='The length of the spatial dimension.')
    parser.add_argument('--w', type=int, default=20, help='The width of the spatial dimension.')
    parser.add_argument('--t', type=int, default=24, help='The length of the time dimension.')
    parser.add_argument('--test_name', type=str, default='temperature', help='Name of the test or dataset.')
    parser.add_argument('--testdata_path', type=str, default='./Datasets/temperature_test.csv',
                        help='Path to the test dataset.')
    parser.add_argument('--model_path', type=str, default='./result_model/result_model.pth',
                        help='Path to the saved model.')
    parser.add_argument('--log_path', type=str, default='./result_model/output_log.log',
                        help='Path to the log file.')
    args = parser.parse_args()

    net = STEI.STEINet(args).to(args.device)
    net.load_state_dict(torch.load(args.model_path))
    net.eval()
    RMSE, MAE = test(net, args)
    logger = Logger(log_file_name=args.log_path, log_level=logging.INFO, logger_name='MyLogger').get_log()
    logger.info(f"Task: {args.test_name}, Sensed ratio: {args.sensed_ratio}, RMSE: {RMSE}, MAE: {MAE}")

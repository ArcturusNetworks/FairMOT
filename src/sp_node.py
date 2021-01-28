from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq


logger.setLevel(logging.INFO)

def run_node(opt):
    logger.info('Initializing centernet tracker...')
    executor = datasets.LoadExecutor(opt.in_port, opt.img_size)

    exec_frame(opt, executor, use_cuda=opt.gpus!=[-1])

# Run with the following command
# python3 sp_node.py mot --in_port 5555 --out_port 5558
#   --load_model ../models/fairmot_dla34.pth --conf_thres 0.4 

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    run_node(opt)
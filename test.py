import argparse
import os
import random
import json
from tqdm import tqdm
from PIL import Image

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from rsgpt.common.config import Config
from rsgpt.common.dist_utils import get_rank
from rsgpt.common.registry import registry
from rsgpt.conversation.conversation import Chat, CONV_VISION

from rsgpt.datasets.builders import *
from rsgpt.models import *
from rsgpt.processors import *
from rsgpt.runners import *
from rsgpt.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--out-path", required=True, help="path to save file.")
    parser.add_argument("--task", required=True, choices=["ic", "vqa"], help="evaluation task, image captioning or visual question answering.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Start Testing
# ========================================
img_dir = 'dataset/RSIEval/images'
imgs = sorted(os.listdir(img_dir))

output_dir = args.out_path
os.makedirs(output_dir, exist_ok=True)

if args.task == 'ic':
    user_message = "Please provide a detailed description of the picture."
    outputs = []
    for i, img in enumerate(tqdm(imgs)):
        raw_image = Image.open(os.path.join(img_dir, img)).convert("RGB")
        image = vis_processor(raw_image).unsqueeze(0).to('cuda:{}'.format(args.gpu_id))
        llm_message = model.generate({"image": image, "prompt": user_message})[0]
     
        print('Input image: {}'.format(img))
        print('User: {}'.format(user_message))
        print('RSGPT response: {}:'.format(llm_message))

        output_dict = {}
        output_dict['image_name'] = img
        output_dict['user_message'] = user_message
        output_dict['output_caption'] = llm_message
        outputs.append(output_dict)

    with open(os.path.join(output_dir, f'rsgpt_caption.txt'), 'w') as f:
        json.dump(outputs, f)
else: # 'vqa'

    with open('dataset/RSIEval/annotations.json') as f:
        data = json.load(f)['annotations']

    outputs = []
    for item in tqdm(data):
        filename = item['filename']
        qa_pairs = item['qa_pairs']
        output_dict = {}
        output_dict['image_name'] = filename
        output_dict['qa_pairs'] = []
        print('Input image: {}'.format(filename))

        raw_image = Image.open(os.path.join(img_dir, filename)).convert("RGB")
        image = vis_processor(raw_image).unsqueeze(0).to('cuda:{}'.format(args.gpu_id))

        for qa in qa_pairs:
            question = qa['question']
            answer = qa['answer']
            if question == '':
                continue
            user_message = question
            llm_message = model.generate({"image": image, "prompt": user_message})[0]

            print('Qustion: {}'.format(user_message))
            print('GT answer: {}'.format(answer))
            print('RSGPT answer: {}:'.format(llm_message))

            output_qa = {}
            output_qa['question'] = question
            output_qa['gt_answer'] = answer
            output_qa['pred_answer'] = llm_message
            output_dict['qa_pairs'].append(output_qa)
        outputs.append(output_dict)

    with open(os.path.join(output_dir, f'rsgpt_vqa.txt'), 'w') as f:
        json.dump(outputs, f)


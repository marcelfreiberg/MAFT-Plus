#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from PIL import Image
import tqdm
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import json

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import ColorMode, random_color
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from maft import add_maskformer2_config, add_fcclip_config
from demo.predictor import VisualizationDemo, DefaultPredictor, OpenVocabVisualizer
from maft.data.datasets.openseg_classes import get_lars_coco_categories_with_prompt_eng

setup_logger(name="fvcore")
logger = setup_logger()

# In[2]:

# Use the prompt-engineered categories from the text file
LARS_CATEGORIES_PROMPT_ENG = get_lars_coco_categories_with_prompt_eng()

# Separate into thing and stuff classes
LARS_THING_CLASSES_PROMPT_ENG = [cat["name"] for cat in LARS_CATEGORIES_PROMPT_ENG if cat["isthing"] == 1]
LARS_STUFF_CLASSES_PROMPT_ENG = [cat["name"] for cat in LARS_CATEGORIES_PROMPT_ENG if cat["isthing"] == 0]

LARS_THING_COLORS = [cat["color"] for cat in LARS_CATEGORIES_PROMPT_ENG if cat["isthing"] == 1]
LARS_STUFF_COLORS = [cat["color"] for cat in LARS_CATEGORIES_PROMPT_ENG if cat["isthing"] == 0]

ODISE_TO_LARS_MAPPING = {
    0: 11,
    1: 12,
    2: 13,
    3: 14,
    4: 15,
    5: 16,
    6: 17,
    7: 19,
    8: 1,
    9: 3,
    10: 5,
}

VAL_IMAGES_PATH = "/data/mfreiberg/datasets/lars/val/images"
TEST_IMAGES_PATH = "/data/mfreiberg/datasets/lars/test/images"
VAL_OUTPUT_PATH = "output/images/val/P3"
TEST_OUTPUT_PATH = "output/images/test"

def build_demo_classes_and_metadata():
    """Build metadata for LARS dataset with prompt engineering"""
    
    # Create metadata with prompt-engineered class names (comma-separated synonyms)
    MetadataCatalog.pop("fcclip_demo_metadata", None)
    demo_metadata = MetadataCatalog.get("fcclip_demo_metadata")
    
    # Set the classes with comma-separated synonyms from the text file
    demo_metadata.thing_classes = LARS_THING_CLASSES_PROMPT_ENG
    demo_metadata.stuff_classes = LARS_THING_CLASSES_PROMPT_ENG + LARS_STUFF_CLASSES_PROMPT_ENG
    
    demo_metadata.thing_colors = LARS_THING_COLORS
    demo_metadata.stuff_colors = LARS_THING_COLORS + LARS_STUFF_COLORS
    
    demo_metadata.stuff_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.stuff_classes))
    }
    demo_metadata.thing_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.thing_classes))
    }

    return demo_metadata


# In[3]:


def draw_panoptic_seg_lars(image, predictions):
    panoptic_seg, segments_info = predictions["panoptic_seg"]
        
    if isinstance(panoptic_seg, torch.Tensor):
       panoptic_seg = panoptic_seg.cpu().numpy()
        
    panoptic_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        
    for segment in segments_info:
        segment_id = segment["id"]
        is_thing = segment["isthing"]
        category_id = ODISE_TO_LARS_MAPPING[segment["category_id"]]
            
        binary_mask = (panoptic_seg == segment_id)
            
        instance_id = segment_id if is_thing else 0
        
        g_channel = instance_id // 256
        b_channel = instance_id % 256
        
        panoptic_mask[binary_mask, 0] = category_id
        panoptic_mask[binary_mask, 1] = g_channel
        panoptic_mask[binary_mask, 2] = b_channel

    return Image.fromarray(np.uint8(panoptic_mask))


# In[4]:


def draw_panoptic_seg(image, predictions, demo_metadata):
    v = OpenVocabVisualizer(image[:, :, ::-1], demo_metadata, instance_mode=ColorMode.IMAGE)

    panoptic_result = v.draw_panoptic_seg(predictions["panoptic_seg"][0].to("cpu"), predictions["panoptic_seg"][1]).get_image()

    return Image.fromarray(np.uint8(panoptic_result))


# In[5]:


cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
add_fcclip_config(cfg)
cfg.merge_from_file("configs/panoptic/eval_lars_coco.yaml")
cfg.merge_from_list(['MODEL.WEIGHTS', "/data/mfreiberg/weights/maftplus/maftp_l_pano.pth"])
cfg.merge_from_list(['MODEL.META_ARCHITECTURE', "MAFT_Plus_DEMO"])
cfg.freeze()


# In[6]:


predictor = DefaultPredictor(cfg)

demo_metadata = build_demo_classes_and_metadata()

predictor.set_metadata(demo_metadata)

# In[9]:


def inference(img_path, output_dir):
    fps_values = []
    image_names = []

    for path in tqdm.tqdm(glob.glob(f"{img_path}/*")):
        image_names.append(os.path.basename(path))
        img_raw = cv2.imread(path)

        start_time = time.time()
        predictions = predictor(img_raw)
        end_time = time.time()

        fps_values.append(1.0 / (end_time - start_time))

        img = draw_panoptic_seg(img_raw, predictions, demo_metadata)
        img_lars = draw_panoptic_seg_lars(img_raw, predictions)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/panoptic", exist_ok=True)
        os.makedirs(f"{output_dir}/lars_format", exist_ok=True)

        img.save(f"{output_dir}/panoptic/{os.path.splitext(os.path.basename(path))[0]}.png")
        img_lars.save(f"{output_dir}/lars_format/{os.path.splitext(os.path.basename(path))[0]}.png")

    fps_stats = {
        "mean_fps": np.mean(fps_values),
        "median_fps": np.median(fps_values),
        "min_fps": np.min(fps_values),
        "max_fps": np.max(fps_values),
        "std_fps": np.std(fps_values),
        "fps_per_image": dict(
            zip(image_names, fps_values)
        ),
    }

    with open(os.path.join(output_dir, "fps_stats.json"), "w") as f:
        json.dump(fps_stats, f, indent=4)


# In[10]:


inference(VAL_IMAGES_PATH, VAL_OUTPUT_PATH)
# inference(TEST_IMAGES_PATH, TEST_OUTPUT_PATH)


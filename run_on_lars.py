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

from maft import add_maskformer2_config, add_fcclip_config
from demo.predictor import VisualizationDemo, DefaultPredictor, OpenVocabVisualizer


# In[2]:

LARS_STUFF_CLASSES = [
    ["Static Obstacle"],
    ["Water"],
    ["Sky"],
]

LARS_STUFF_CLASSES_EXTENDED = [
    ["Static Obstacle", "Fixed Object", "Immovable Object", "Barrier", "Structure", "Terrain", "Ground", "Land", "Forest", "Trees", "Vegetation", "Plants", "Grass", "Bush", "Foliage", "Beach", "Shore", "Coast", "Sand", "Rocks", "Cliff", "Stone", "Boulder", "Background", "Environment", "Surroundings", "Landscape", "Dock", "Pier", "Jetty", "Building", "Construction"],
    ["Water", "Sea", "Ocean", "Lake", "River", "Pond", "Fluid", "Aquatic surface", "Waterway", "Stream", "Waves", "Liquid"],
    ["Sky", "Clouds", "Atmosphere", "Heavens", "Air", "Horizon", "Celestial", "Firmament"],
]

LARS_STUFF_COLORS = [
    [247, 195, 37],
    [41, 167, 224],
    [90, 75, 164],
]

LARS_THING_CLASSES = [
    ["Boat/ship", "Boat with person"],
    ["Row boats"],
    ["Paddle board"],
    ["Buoy"],
    ["Swimmer"],
    ["Animal"],
    ["Float"],
    ["Other"],
]

LARS_THING_COLORS = [
    [255, 87, 51],    # Boat/ship - bright orange-red
    [50, 205, 50],    # Row boats - lime green  
    [255, 0, 255],    # Paddle board - magenta
    [255, 215, 0],    # Buoy - gold
    [0, 128, 128],    # Swimmer - teal
    [139, 69, 19],    # Animal - saddle brown
    [220, 20, 60],    # Float - crimson
    [169, 169, 169],  # Other - dark gray
]

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

def build_demo_classes_and_metadata(vocab, label_list):
    extra_classes = []

    if vocab:
        for words in vocab.split(";"):
            extra_classes.append([word.strip() for word in words.split(",")])
    extra_colors = [random_color(rgb=True, maximum=1) for _ in range(len(extra_classes))]

    demo_thing_classes = extra_classes
    demo_stuff_classes = []
    demo_thing_colors = extra_colors
    demo_stuff_colors = []

    if "LARS" in label_list:
        demo_thing_classes += LARS_THING_CLASSES
        demo_stuff_classes += LARS_STUFF_CLASSES
        demo_thing_colors += LARS_THING_COLORS
        demo_stuff_colors += LARS_STUFF_COLORS
    if "LARS_EXTENDED" in label_list:
        demo_thing_classes += LARS_THING_CLASSES
        demo_stuff_classes += LARS_STUFF_CLASSES_EXTENDED
        demo_thing_colors += LARS_THING_COLORS
        demo_stuff_colors += LARS_STUFF_COLORS

    MetadataCatalog.pop("fcclip_demo_metadata", None)
    demo_metadata = MetadataCatalog.get("fcclip_demo_metadata")
    demo_metadata.thing_classes = [c[0] for c in demo_thing_classes]
    demo_metadata.stuff_classes = [
        *demo_metadata.thing_classes,
        *[c[0] for c in demo_stuff_classes],
    ]
    demo_metadata.thing_colors = demo_thing_colors
    demo_metadata.stuff_colors = demo_thing_colors + demo_stuff_colors
    demo_metadata.stuff_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.stuff_classes))
    }
    demo_metadata.thing_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.thing_classes))
    }

    demo_classes = demo_thing_classes + demo_stuff_classes

    return demo_classes, demo_metadata


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
cfg.merge_from_file("configs/panoptic/eval.yaml")
cfg.merge_from_list(['MODEL.WEIGHTS', "/data/mfreiberg/weights/maftplus/maftp_l_pano.pth"])
cfg.merge_from_list(['MODEL.META_ARCHITECTURE', "MAFT_Plus_DEMO"])
cfg.freeze()


# In[6]:


predictor = DefaultPredictor(cfg)

demo_classes, demo_metadata = build_demo_classes_and_metadata("", ["LARS_EXTENDED"]) # Class config. Options: "LARS", "LARS_EXTENDED", ""
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


inference("/data/mfreiberg/datasets/lars/val/images", "/data/mfreiberg/predictions/maftplus/val")


# In[11]:


inference("/data/mfreiberg/datasets/lars/test/images", "/data/mfreiberg/predictions/maftplus/test")


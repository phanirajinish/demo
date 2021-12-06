# Core Packages
import streamlit as st
import cv2
import numpy as np
import os
import os
import time
import datetime
import json
import random
import copy
import logging
import matplotlib.pyplot as plt
import cv2
import numpy as np
from glob import glob
import detectron2
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.engine.hooks import HookBase
from detectron2.structures import BoxMode
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import log_every_n_seconds
from detectron2 import model_zoo
from detectron2.config import get_cfg,CfgNode
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator, inference_on_dataset,inference_context
import detectron2.utils.comm as comm
from detectron2.data.datasets import register_coco_instances
from PIL import Image

setup_logger()
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

classes = ['background',
 'back_bumper',
 'back_glass',
 'back_left_door',
 'back_left_light',
 'back_right_door',
 'back_right_light',
 'front_bumper',
 'front_glass',
 'front_left_door',
 'front_left_light',
 'front_right_door',
 'front_right_light',
 'hood',
 'left_mirror',
 'right_mirror',
 'tailgate',
 'trunk',
 'wheel']

@st.cache
def get_predictor():
    thresh = 0.7
    MODEL = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    WEIGHT_PATH = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    model_path = "model_final.pth"


    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL) 
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    cfg.DATASETS.TEST =  ("data_test",)
    cfg.TEST.DETECTIONS_PER_IMAGE = 20
    cfg.TEST.EVAL_PERIOD = 100
    cfg.DATALOADER.NUM_WORKERS = 20
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.BASE_LR = 0.001   
    cfg.SOLVER.MAX_ITER = 2500
    cfg.SOLVER.WEIGHT_DECAY =  0.0001
    cfg.SOLVER.CHECKPOINT_PERIOD = 500

    cfg.MODEL.WEIGHTS =  model_path   
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh

    predictor = DefaultPredictor(cfg)

    return predictor

@st.cache
def load_image(img):
    im = cv2.imread(img)
    return im


def main():

    """CAR PARTS DETECTION"""

    st.title("CAR PARTS INSTANCE SEGMENTATION APP")
    # st.text("For M-Tech in Data Science, BITS Pilani")
    st.text("Rajanish Chava")

    activties = ["Segmentation","About"]
    choice = st.sidebar.selectbox("select activity", activties)

    if choice == "Segmentation":
        st.subheader("Car Parts Instance Segmentation")

        image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

        col1 , col2 = st.columns(2)

        if image_file is not None:
            print("image_file",image_file)
            im = Image.open(image_file)  
            col1.image(im)
            # st.image(im)

        if st.button("DETECT CAR PARTS"):
            start_time = time.time()

            open_cv_image = np.array(im) 
            predictor = get_predictor()
            outputs = predictor(open_cv_image)
            MetadataCatalog.get("data_test").set(thing_classes=classes)
            v = Visualizer(open_cv_image,
                   metadata=MetadataCatalog.get("data_test"), 
                   scale=1, 
                   instance_mode=ColorMode.IMAGE_BW)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1]
            im_pil = Image.fromarray(out)
            col2.image(im_pil)
            total_time = (time.time() - start_time)
            print(f"---FPS--- %s seconds ---{1/total_time}") 
            st.write(f"---Frames Per Sec--- %s seconds ---{1/total_time}")


    elif choice == "About":
        st.subheader("About")
        st.text("This mobile app segments car parts.\nPlease upload an image of a car")

if __name__=="__main__":
    main()
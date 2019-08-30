import mrcnn.model as modellib
import imgaug
import mrcnn.utils as utils
import os

from FashionConfig import FashionConfig
from FashionDataset import FashionDataset

COCO_MODEL_PATH = "mask_rcnn_coco.h5"
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

config = FashionConfig()
train_dataset = FashionDataset()
val_dataset = FashionDataset(mode='validation')
# config.display()
# print(train_dataset.class_info)
# print(len(train_dataset.image_info))

train_dataset.prepare()
val_dataset.prepare()

model = modellib.MaskRCNN(mode='training', config=config, model_dir='results')
#model.load_weights(model.find_last(), by_name=True)

model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])

augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 2.0)),
                    imgaug.augmenters.PiecewiseAffine(scale=(0.01, 0.05)),
                    imgaug.augmenters.Add((-25, 25)),
                    imgaug.augmenters.JpegCompression((50, 80)),
                    imgaug.augmenters.Affine(rotate=(-90, 90))
                ])

config.WEIGHT_DECAY = 0.0001

model.train(train_dataset, val_dataset, learning_rate=config.LEARNING_RATE, epochs=25, layers='heads', augmentation=augmentation)
# model.train(train_dataset, val_dataset, learning_rate=config.LEARNING_RATE / 10, epochs=60, layers='4+', augmentation=augmentation)
# model.train(train_dataset, val_dataset, learning_rate=config.LEARNING_RATE / 10, epochs=120, layers='all', augmentation=augmentation)
import mrcnn.model as modellib
import imgaug

from FashionConfig import FashionConfig
from FashionDataset import FashionDataset

config = FashionConfig()
train_dataset = FashionDataset()
val_dataset = FashionDataset(mode='validation')
# config.display()
# print(train_dataset.class_info)
# print(len(train_dataset.image_info))

train_dataset.prepare()
val_dataset.prepare()

model = modellib.MaskRCNN(mode='training', config=config, model_dir='results')
model.load_weights(model.find_last(), by_name=True)

augmentation = imgaug.augmenters.Sometimes(p=0.5, then_list=[
    imgaug.augmenters.Fliplr(0.5),
    imgaug.augmenters.Affine()
    ])
config.LEARNING_RATE = 0.001
config.LEARNING_MOMENTUM = 0.5
config.WEIGHT_DECAY = 0.001
model.train(train_dataset, val_dataset, learning_rate=config.LEARNING_RATE, epochs=40, layers='4+', augmentation=augmentation)
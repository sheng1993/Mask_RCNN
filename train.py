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
model.load_weights('mask_rcnn_fashionbot_0020.h5', by_name=True)

augmentation = imgaug.augmenters.Sometimes(p=0.5, then_list=[
    imgaug.augmenters.Fliplr(0.5)
    ])
config.LEARNING_RATE = 0.0001
model.train(train_dataset, val_dataset, learning_rate=config.LEARNING_RATE, epochs=20, layers='all', augmentation=augmentation)
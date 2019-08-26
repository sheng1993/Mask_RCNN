import mrcnn.model as modellib

from FashionConfig import FashionConfig
from FashionDataset import FashionDataset

config = FashionConfig()
train_dataset = FashionDataset()
val_dataset = FashionDataset(mode='validation')
config.display()
print(train_dataset.class_info)
# print(len(train_dataset.image_info))

train_dataset.prepare()
val_dataset.prepare()

model = modellib.MaskRCNN(mode='training', config=config, model_dir='results')
model.load_weights(model.get_imagenet_weights(), by_name=True)

model.train(train_dataset, val_dataset, learning_rate=config.LEARNING_RATE, epochs=20, layers='heads')
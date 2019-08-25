import mrcnn.model as modellib

from FashionConfig import FashionConfig
from FashionDataset import FashionDataset

config = FashionConfig()
train_dataset = FashionDataset()
#config.display()
# print(train_dataset.class_info)
print(len(train_dataset.image_info))

model = modellib.MaskRCNN(mode='training', config=config, model_dir='results')
model.load_weights(model.get_imagenet_weights(), by_name=True)

#model.train()
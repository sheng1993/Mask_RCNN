from FashionConfig import FashionConfig
from FashionDataset import FashionDataset

config = FashionConfig()
#config.display()
train_dataset = FashionDataset()
print(train_dataset.class_info)
print(train_dataset.image_info)
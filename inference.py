import mrcnn.model as modellib
import skimage
import matplotlib.pyplot as plt

from mrcnn import visualize, utils
from FashionConfig import InferenceConfig
from FashionDataset import FashionDataset

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

config = InferenceConfig()

train_dataset = FashionDataset()
train_dataset.prepare()

model = modellib.MaskRCNN(mode='inference', config=config, model_dir='results')
model.load_weights('imaterialist_mask_rcnn_fashionbot_0020.h5', by_name=True)

while True:
    img_path = input('Image = ')
    image = skimage.io.imread(img_path)

    image, _, _, _, _ = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)

    results = model.detect([image], verbose=1)
    r = results[0]

    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                train_dataset.class_names, r['scores'],
                                title="Predictions")
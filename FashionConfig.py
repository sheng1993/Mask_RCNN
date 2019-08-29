from mrcnn.config import Config

class FashionConfig(Config):
    NAME = 'FashionBot'

    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    NUM_CLASSES = 16

    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

class InferenceConfig(FashionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

if __name__ == '__main__':
    config = FashionConfig()
    config.display()
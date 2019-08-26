from mrcnn.config import Config

class FashionConfig(Config):
    NAME = 'FashionBot'

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 16

    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

if __name__ == '__main__':
    config = FashionConfig()
    config.display()
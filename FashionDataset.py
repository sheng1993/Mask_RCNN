from mrcnn.utils import Dataset

import os
import pickle
import numpy as np

class FashionDataset(Dataset):

    def __init__(self, mode='training', class_map=None):
        super().__init__(class_map=class_map)

        with open('class_info.pickle', 'rb') as f:
            conf = pickle.load(f)
        
        self.class_info.extend(conf)


        with open('image_info.pickle', 'rb') as f:
            image_info = pickle.load(f)
        
        self.image_info = image_info[:45000] if mode == 'training' else image_info[45000:]
    
    def load_image(self, image_id):
        return super().load_image(image_id)

    def load_mask(self, image_id: str):
        image_id = image_id.split('.')[0]
        with open(os.path.join('data', image_id + '.txt'), 'r') as f:
            lines = f.readlines()
            width, height = lines[0].split(' ')
            width, height = int(width), int(height)

            masks = np.zeros((height, width, len(lines) - 1))
            for i in range(len(lines) - 1):
                mask = self.get_partial_image_mask(width, height, lines[i + 1])
                masks[:, :, i] = mask
        
        class_ids = np.load(os.path.join('masks', image_id + '_classes.npy'))
        return masks, class_ids

    
    def get_partial_image_mask(self, seg_width, seg_height, encoded_pixels):
        seg_img = np.full(seg_width*seg_height, 0, dtype=np.uint8)

        pixel_list = list(map(int, encoded_pixels.split(" ")))
        for i in range(0, len(pixel_list), 2):
            start_index = pixel_list[i] - 1
            index_len = pixel_list[i+1] - 1
            seg_img[start_index:start_index+index_len] = 1
            
        seg_img = seg_img.reshape((seg_height, seg_width), order='F')
        return seg_img
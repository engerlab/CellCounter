# https://huggingface.co/ybelkada/segment-anything
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import build_sam, SamPredictor 


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

images = os.listdir(os.path.join('..', 'img'))
img_path = os.path.join('..', 'img')
out_path = os.path.join('.', 'output_b')
checkpoint_path = os.path.join('.', 'sam_vit_b_01ec64.pth')

for image in tqdm(images):
    img = cv2.imread(os.path.join(img_path, image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device='cuda')
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(img)

    plt.figure(figsize=(20,20))
    plt.imshow(img)
    show_anns(masks)
    plt.axis('off')
    plt.savefig(os.path.join(out_path, image[:-4] + '_out.jpg'))

    plt.close()
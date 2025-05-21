import os
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

img_path = os.path.join('.', 'img', 'Sample_1-1.jpg')
checkpoint_path = os.path.join('.', 'sam_vit_h_4b8939.pth')

sam = sam_model_registry["default"](checkpoint=checkpoint_path)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(img_path)

plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('off')
plt.savefig(os.path.join('.', 'output', 'out_Sample_1-1.jpg'))

plt.close()
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import warnings
import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2
class SAM2_MG:
    def __init__(self):
        sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        self.sam2 = build_sam2(model_cfg, sam2_checkpoint, device="cuda", apply_postprocessing=False)

        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam2
            #points_per_side=64,
            #points_per_batch=128,
            #pred_iou_thresh=0.7,
            #stability_score_thresh=0.92,
            #stability_score_offset=0.7,
            #crop_n_layers=1,
            #box_nms_thresh=0.7,
            #crop_n_points_downscale_factor=2,
            #min_mask_region_area=25.0,
            #use_m2m=True,
        )
    
    def predict(self, img):
        masks = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            masks = self.mask_generator.generate(img)
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        mask_img = np.ones((sorted_masks[0]['segmentation'].shape[0], sorted_masks[0]['segmentation'].shape[1], 3))
        mask_img[:, :] = 0
        for mask in sorted_masks:
            m = mask['segmentation']
            color_mask = np.concatenate([np.random.random(3)])
            mask_img[m] = color_mask 
            
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(mask_img, contours, -1, (0, 0, 1), thickness=1)

        return sorted_masks, mask_img
    def __str__(self):
        out_str = f"SAM2: {self.sam2.device=}, "
        #print(out_str)
        #print(f"{dir(self.mask_generator)}")
        #out_str += f"{self.mask_generator.device=}"
        return  out_str
    def __repr__(self):
        return self.__str__()




if __name__ == "__main__":
    sam = SAM2_MG()
    print(sam)
    with open("./custom_dataset/one on two/top_view.pkl", "rb") as file:
        rgb_img, depth_img, pose, K, depth_scale = pickle.load(file)
    masks, mask_img = sam.predict(rgb_img)
    print(f"{len(masks)}")
    for k,v in masks[0].items():
        if k == "segmentation":
            print(f"{k}:{v.shape}\n")
        else:
            print(f"{k}:{v}\n")
    fig, axes = plt.subplots(nrows=2, figsize=(20,20))
    axes[0].imshow(rgb_img)
    axes[1].imshow(mask_img)
    plt.show()

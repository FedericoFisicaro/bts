from PIL import Image
import os
import random
import numpy as np

def fonc(idx):
        
        mode = 'train'
        filenames_file = "../train_test_inputs/personal_train_files_with_gt.txt"
        data_path =  "/home/FisicaroF/UmonsIndoorDataset/dataset"
        gt_path = "/home/FisicaroF/UmonsIndoorDataset/dataset"
        do_random_rotate = True
        degree = 2.5

        with open(filenames_file, 'r') as f:
            filenames = f.readlines()

        sample_path = filenames[idx]
        print(sample_path)
        focal = float(sample_path.split()[2])

        if mode == 'train':
    
            image_path = os.path.join(data_path, "./" + sample_path.split()[0])
            depth_path = os.path.join(gt_path, "./" + sample_path.split()[1])
    
            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)
            
            # if self.args.do_kb_crop is True:
            #     height = image.height
            #     width = image.width
            #     top_margin = int(height - 352)
            #     left_margin = int((width - 1216) / 2)
            #     depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            #     image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            
            # To avoid blank boundaries due to pixel registration
            # if self.args.dataset == 'nyu':
            #     depth_gt = depth_gt.crop((43, 45, 608, 472))
            #     image = image.crop((43, 45, 608, 472))
    
            if do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * degree
                image = rotate_image(image, random_angle)
                depth_gt = rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
            
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

           
            depth_gt = depth_gt / 1000.0
           

            image, depth_gt = random_crop(image, depth_gt, 544, 1280)
            image, depth_gt = train_preprocess(image, depth_gt)
            sample = {'image': image, 'depth': depth_gt, 'focal': focal}
        
        return sample

def rotate_image(image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

def random_crop(img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

def train_preprocess(image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
    
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = augment_image(image)
    
        return image, depth_gt
    
def augment_image(image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma


        brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug


if __name__ == '__main__':
    for i in range(55936):
        print(i)
        fonc(i)


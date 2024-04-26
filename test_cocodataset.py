import json
import cv2
import pytest
import numpy as np
from pathlib import Path
from cocodataset import CocoDataset

np.random.seed(42)


class VisualizeCocoDataset:
    def __init__(self, coco_dir_path: Path, split: str):
        """
        Constructor to visualize skeleton from keypoints on coco images
        :param coco_dir_path: path to the coco main directory
        :param split: If the dataset is 'train' or 'val'
        """
        self.split = split
        self.coco_dir_path = coco_dir_path
        self.coco_dataset = CocoDataset(coco_dir_path, split)
        self.skeleton_structure = self.coco_dataset.get_skeleton_structure()

    def visualize_skeleton_from_keypoints(self, anno_data: dict):
        """
        Visualize skeleton on image from annotated keypoints data
        :param anno_data: annotated keypoints data
        :return:
        """
        image_np = self.image_read(anno_data['meta_info']['file_name'])
        image_shape = (anno_data['meta_info']['height'], anno_data['meta_info']['width'])
        skeleton_mask = self.get_skeleton(anno_data['body_parts'], image_shape)
        if len(image_np.shape) == 2:
            image_np = np.tile(image_np[:, :, None], 3)
        if len(image_np.shape) == 3 and image_np.shape[2] == 1:
            image_np = np.tile(image_np, 3)

        skeleton_image = ((image_np.astype(np.float32) * 0.5) +
                          (skeleton_mask.astype(np.float32) * 0.5)).astype(np.uint8)
        merged_image = np.concatenate([image_np, skeleton_mask, skeleton_image], axis=1)
        merged_image = cv2.resize(merged_image, (480 * 3, 480))
        print(f"image id: {anno_data['image_id']}")
        cv2.imshow(f"original image-->skeleton mask-->skeleton image (image id: "
                   f"{anno_data['image_id']})", merged_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def image_read(self, filename: str) -> np.ndarray:
        """
        Read coco image from the given filename
        :param filename: coco image filename to read
        :return: numpy array of the image
        """
        image_path = self.coco_dir_path / 'images' / f'{self.split}2017' / filename
        assert image_path.is_file(), f'Image file is not available in the path: {image_path}!'
        image_np = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        assert image_np is not None, f'Error reading image: {image_path}'
        return image_np

    def get_skeleton(self, keypoint_anno: list, image_shape: tuple,
                     include_occluded: bool = False) -> np.ndarray:
        """
        Draws skeleton mask from keypoints
        :param keypoint_anno: list containing instance keypoint annotations
        :param image_shape: image shape as height x width
        :param include_occluded: boolean to indicate whether occluded keypoints should be
        included or not
        :return: skeleton drawn on black background
        """
        skeleton_mask = np.zeros((*image_shape, 3), dtype=np.uint8)
        for ins_keypoint_anno in keypoint_anno:
            for each_link in self.skeleton_structure:
                kp1_name = self.coco_dataset.person_keypoints[each_link[0] - 1]
                kp2_name = self.coco_dataset.person_keypoints[each_link[1] - 1]
                kp1_dict = ins_keypoint_anno[kp1_name]
                kp2_dict = ins_keypoint_anno[kp2_name]

                visibility_thresh = 1
                if include_occluded:
                    visibility_thresh = 0

                if kp1_dict['visibility'] > visibility_thresh:
                    skeleton_mask = cv2.circle(skeleton_mask, (kp1_dict['x'], kp1_dict['y']), 2,
                                               (0, 0, 255), 3)
                if kp2_dict['visibility'] > visibility_thresh:
                    skeleton_mask = cv2.circle(skeleton_mask, (kp2_dict['x'], kp2_dict['y']), 2,
                                               (0, 0, 255), 3)
                if kp1_dict['visibility'] > visibility_thresh and \
                        kp2_dict['visibility'] > visibility_thresh:
                    skeleton_mask = cv2.line(skeleton_mask, (kp1_dict['x'], kp1_dict['y']),
                                             (kp2_dict['x'], kp2_dict['y']), (120, 255, 0), 2)

        return skeleton_mask


class TestCocoDataset:
    """
    Test class for testing coco dataset
    """

    @pytest.mark.repeat(50)
    def test_Visualize_output(self):
        """
        Test case to visualize the implementation by obtaining data directly from the function
        :return:
        """
        coco_dir_path = Path(f'D:\DataSets\coco_data')
        split = 'val'
        viz_coco_dataset = VisualizeCocoDataset(coco_dir_path, split)
        all_image_ids = viz_coco_dataset.coco_dataset.get_all_image_ids()
        dataset_len = len(viz_coco_dataset.coco_dataset)
        random_index = np.random.randint(0, dataset_len)
        image_id = all_image_ids[random_index]

        anno_data = viz_coco_dataset.coco_dataset.get_anno_dict_from_image_id(image_id)
        viz_coco_dataset.visualize_skeleton_from_keypoints(anno_data)

    @pytest.mark.repeat(50)
    def test_Visualise_from_json(self):
        """
        Test case to visualize the implementation by reading data from json file.
        :return:
        """
        split = 'val'
        coco_dir_path = Path(f'D:\DataSets\coco_data')
        json_file_path = coco_dir_path / f'coco_keypoint_{split}2017.json'
        assert json_file_path.is_file(), f'Json file not found in the path: {json_file_path}'

        with open(json_file_path, 'r') as json_file_read:
            json_anno_data = json.load(json_file_read)
        viz_coco_dataset = VisualizeCocoDataset(coco_dir_path, split)
        dataset_len = len(json_anno_data)
        random_index = np.random.randint(0, dataset_len)
        anno_data = json_anno_data[random_index]

        viz_coco_dataset.visualize_skeleton_from_keypoints(anno_data)

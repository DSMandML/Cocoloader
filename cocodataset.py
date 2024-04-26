import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pycocotools.coco import COCO


class CocoDataset:
    def __init__(self, coco_dir_path: Path, split: str):
        """
        Constructor for converting coco keypoint annotations to JSON format as below:
        [
        {
            "image_id": "123",
            "meta_info": {...},
            "body_parts": [{
                  "nose": {
                  "x": 3,
                  "y": 4,
                  "z": 14,
                  "visibility": 0
                  },
            ...
            },
            {
            ...
            }]
        },
        {
        ...
        },
        ...
        ]
        :param coco_dir_path: path to the coco main directory
        :param split: If the dataset is 'train' or 'val'
        """
        assert split in ('train', 'val'), \
            f"Invalid spliot. Split should be either 'val' or 'train'"

        self.split = split
        self.coco_dir_path = coco_dir_path
        self.annotation_path = coco_dir_path / 'annotations' / f'person_keypoints_{split}2017.json'
        self.check_paths()
        self.keypoint_anno_data = COCO(self.annotation_path)
        self.person_id = self.keypoint_anno_data.getCatIds(catNms=['person'])
        self.person_keypoints = self.keypoint_anno_data.loadCats(self.person_id)[0]['keypoints']
        self.all_image_ids = self.keypoint_anno_data.getImgIds()

    def check_paths(self):
        """
        Check if different image directories and the annotation files exists
        """
        assert self.coco_dir_path.is_dir(), \
            f"Coco directory does not exist in the path: {self.coco_dir_path}"
        assert self.annotation_path.is_file(), \
            f"Coco annotation file does not exists in the path: {self.annotation_path}"

    def get_annotation_in_json(self) -> list:
        """
        Generates json format output of coco keypoint annotations
        :return: keypoint annotations along with metadata for all the images in json format
        """
        all_anno_dict_list = []
        for image_id in tqdm(self.all_image_ids):
            each_anno_dict = self.get_anno_dict_from_image_id(image_id)
            all_anno_dict_list.append(each_anno_dict)
        return all_anno_dict_list

    def get_anno_dict_from_image_id(self, image_id: int) -> dict:
        """
        Extract keypoint annotation data of an image from an image id
        :param image_id: coco image id
        :return: dictionary containing keypoint annotation data
        """
        image_metadata = self.keypoint_anno_data.loadImgs([image_id])
        anno_ids = self.keypoint_anno_data.getAnnIds(imgIds=[image_id], catIds=self.person_id,
                                                     iscrowd=False)
        each_anno_dict = {'image_id': str(image_id),
                          'meta_info': self.get_metadata_dict(image_metadata[0], anno_ids),
                          'body_parts': self.get_bodyparts_data(anno_ids)}
        return each_anno_dict

    def get_metadata_dict(self, metadata_dict: dict, anno_ids: list) -> dict:
        """
        Creates a dictionary with metadata information
        :param metadata_dict: coco image metadata information
        :param anno_ids: instance annotation ids
        :return: dictionary containing metadata
        """
        data_dict = metadata_dict.copy()
        if 'id' in data_dict:
            del data_dict['id']
        data_dict['annotation_ids'] = anno_ids
        return data_dict

    def get_bodyparts_data(self, anno_ids: list) -> list:
        """
        Creates list of dictionary for different instances containing keypoints of different
        body parts
        :param anno_ids: keypoint annotation id
        :return: list of dictionary containing keypoints grouped into body parts
        """
        bodyparts_dict = []
        anno_data = self.keypoint_anno_data.loadAnns(anno_ids)
        for each_anno_data in anno_data:
            ins_bodyparts_dict = {}
            ins_keypoints = np.array(each_anno_data['keypoints'])\
                .reshape(len(self.person_keypoints), -1)
            for kp_name, kp_value in zip(self.person_keypoints, ins_keypoints):
                if kp_value.shape[0] == 3:
                    z_val = int(0)
                elif kp_value.shape[0] == 4:
                    z_val = int(kp_value[2])
                else:
                    raise NotImplementedError
                ins_bodyparts_dict[kp_name] = {'x': int(kp_value[0]),
                                               'y': int(kp_value[1]),
                                               'z': z_val,
                                               'visibility': int(kp_value[2])}
            bodyparts_dict.append(ins_bodyparts_dict)
        return bodyparts_dict

    def get_all_image_ids(self) -> list:
        """
        :return: list of image ids
        """
        return self.all_image_ids

    def get_skeleton_structure(self) -> list:
        """
        :return: list of indices indicating the skeleton connectivity
        """
        return self.keypoint_anno_data.loadCats(self.person_id)[0]['skeleton']

    def __len__(self) -> int:
        """
        :return: length of the dataset.
        """
        return len(self.all_image_ids)


def main():
    default_coco_dir_path = f'D:\DataSets\coco_data'
    default_split = 'val'

    parser = argparse.ArgumentParser(
        description="Transform COCO keypoint annotations to the custom json format")
    parser.add_argument("--path", type=str, default=default_coco_dir_path,
                        help="Path to the main coco directory")
    parser.add_argument("--split", type=str, default=default_split,
                        help="Coco split 'train' or 'val' to be used.")
    args = parser.parse_args()
    print('Arguments:')
    print(vars(args))

    coco_dataset = CocoDataset(Path(args.path), args.split)

    print('Execution started!')
    start_time = time.time()

    json_data = coco_dataset.get_annotation_in_json()
    with open(Path(args.path) / f'coco_keypoint_{args.split}2017.json', 'w') as f:
        json.dump(json_data, f)

    stop_time = time.time()
    print('Execution completed!')
    print(f'Time of execution: {np.round(stop_time - start_time, 2)} Sec')


if __name__ == '__main__':
    main()

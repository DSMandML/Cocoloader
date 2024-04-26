import cv2
import time
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pycocotools.coco import COCO


class CocoInstanceDataset:
    def __init__(self, coco_dir_path: Path, split: str, expand_bbox: float = 0.2,
                 filter_on_min_area: float = 20.0 * 20.0, filter_on_min_keypoints: int = 3,
                 always_crop: bool = False):
        """
        Constructor for converting coco keypoint annotations of every instance to JSON format as
        below:
        [
        {
            "image_id": "123",
            "meta_info": {...},
            "body_parts": {
                  "nose": {
                  "x": 3,
                  "y": 4,
                  "z": 14,
                  "visibility": 0
                  },
            ...
            }
        },
        {
        ...
        },
        ...
        ]
        :param coco_dir_path: path to the coco main directory
        :param split: If the dataset is 'train' or 'val'
        :param expand_bbox: Expand the bounding box by x% before cropping
        :param filter_on_min_area: Ignore the instances that has area less than filter_on_min_area
        :param filter_on_min_keypoints: Ignore the instances that has number of labelled
        keypoints less than filter_on_min_area
        :param always_crop: boolean value to indicate if cropping based on bounding box is
        required even if there is only single instance in an image
        """
        assert split in ('train', 'val'), f"Invalid split. Split should be either 'val' or 'train'"
        assert 0.0 <= expand_bbox <= 1.0, f'Value error! expand_bbox must be between 0.0 and 1.0'
        self.split = split
        self.always_crop = always_crop
        self.coco_dir_path = coco_dir_path
        self.expand_bbox = expand_bbox
        self.filter_on_min_area = filter_on_min_area
        self.filter_on_min_keypoints = filter_on_min_keypoints
        self.image_dir_path = coco_dir_path / 'images' / f'{split}2017'
        self.annotation_path = coco_dir_path / 'annotations' / f'person_keypoints_{split}2017.json'
        self.check_paths()
        self.save_image_dir = self.coco_dir_path / 'instance_images' / f'{split}2017'
        self.save_image_dir.mkdir(exist_ok=True, parents=True)

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
        assert self.image_dir_path.is_dir(), \
            f"Coco image directory does not exist in the path: {self.image_dir_path}"
        assert self.annotation_path.is_file(), \
            f"Coco annotation file does not exists in the path: {self.annotation_path}"

    def get_annotation_in_json(self) -> list:
        """
        Generates json format output of coco keypoint instance annotations
        :return: keypoint annotations along with metadata for all the instances of all the images
        in json format
        """
        all_anno_dict_list = []
        for image_id in tqdm(self.all_image_ids):
            anno_dict_list = self.get_anno_dict_from_image_id(image_id)
            all_anno_dict_list.extend(anno_dict_list)
        return all_anno_dict_list

    def get_anno_dict_from_image_id(self, image_id: int) -> list:
        """
        Extract instance keypoint annotation data of an image from an image id
        :param image_id: coco image id
        :return: dictionary containing instance keypoint annotation data
        """
        image_metadata = self.keypoint_anno_data.loadImgs([image_id])[0]
        image_filename = image_metadata['file_name']
        image_np = self.image_read(image_filename)
        anno_ids = self.keypoint_anno_data.getAnnIds(imgIds=[image_id], catIds=self.person_id,
                                                     iscrowd=False)
        anno_data = self.keypoint_anno_data.loadAnns(anno_ids)

        anno_dict_list = []
        if len(anno_data) == 0:
            ins_idx = 1
            ins_image_filename = self.save_image(image_np, image_filename, ins_idx)
            ins_image_shape = [image_metadata['height'], image_metadata['width']]

            ins_image_id = int(ins_image_filename.split('.')[0])
            ins_meta_info = self.get_metadata_dict(image_metadata, ins_image_shape,
                                                   ins_image_filename, dict())
            dummy_anno_data = {'keypoints': [0] * len(self.person_keypoints) * 3}
            ins_bodyparts_info = self.get_bodyparts_data(dummy_anno_data)

            ins_anno_dict = {'image_id': ins_image_id,
                             'meta_info': ins_meta_info,
                             'body_parts': ins_bodyparts_info}
            anno_dict_list.append(ins_anno_dict)

        elif len(anno_data) == 1 and not self.always_crop:
            ins_idx = 1
            ins_image_filename = self.save_image(image_np, image_filename, ins_idx)
            ins_image_shape = [image_np.shape[0], image_np.shape[1]]

            ins_image_id = int(ins_image_filename.split('.')[0])
            ins_meta_info = self.get_metadata_dict(image_metadata, ins_image_shape,
                                                   ins_image_filename, anno_data[0])
            ins_bodyparts_info = self.get_bodyparts_data(anno_data[0])

            ins_anno_dict = {'image_id': ins_image_id,
                             'meta_info': ins_meta_info,
                             'body_parts': ins_bodyparts_info}
            anno_dict_list.append(ins_anno_dict)

        else:
            ins_idx = 0
            for ins_anno_data in anno_data:
                if ins_anno_data['num_keypoints'] < self.filter_on_min_keypoints or \
                        ins_anno_data['area'] < self.filter_on_min_area:
                    continue
                ins_idx += 1
                new_ins_anno_data = self.get_instance_annotation(ins_anno_data, image_metadata)
                ins_image_np = self.get_image_crop(image_np, new_ins_anno_data)
                ins_image_filename = self.save_image(ins_image_np, image_filename, ins_idx)
                ins_image_shape = new_ins_anno_data['bbox'][-1:-3:-1]
                ins_image_id = int(ins_image_filename.split('.')[0])

                ins_meta_info = self.get_metadata_dict(image_metadata, ins_image_shape,
                                                       ins_image_filename, ins_anno_data)
                ins_bodyparts_info = self.get_bodyparts_data(new_ins_anno_data)

                ins_anno_dict = {'image_id': ins_image_id,
                                 'meta_info': ins_meta_info,
                                 'body_parts': ins_bodyparts_info}
                anno_dict_list.append(ins_anno_dict)
        return anno_dict_list

    def image_read(self, filename: str) -> np.ndarray:
        """
        Read coco image from the given filename
        :param filename: coco image filename to read
        :return: numpy array of the image
        """
        image_path = self.image_dir_path / filename
        assert image_path.is_file(), f'Image file is not available in the path: {image_path}!'
        image_np = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        assert image_np is not None, f'Error reading image: {image_path}'
        return image_np

    def save_image(self, image_np: np.ndarray, filename: str, ins_idx: int) -> str:
        """
        Save image into the directory
        :param image_np: image to be saved as numpy array
        :param filename: filename for image
        :param ins_idx: index which is appended to filename as 2 digit string in the beginning
        :return: stem part of the saved image filename
        """
        img_stem, img_suffix = filename.split('.')
        save_file_path = self.save_image_dir / f'{img_stem}{ins_idx:02d}.{img_suffix}'
        cv2.imwrite(str(save_file_path), image_np)
        return save_file_path.parts[-1]

    def get_image_crop(self, image_np: np.ndarray, each_anno_data: dict) -> np.ndarray:
        """
        Crop the input image based on bounding box
        :param image_np: image to be cropped
        :param each_anno_data: annotation containing bounding box information
        :return: cropped image as numpy array
        """
        x1, y1, width, height = each_anno_data['bbox']
        x2, y2 = x1 + width, y1 + height
        return image_np[y1:y2, x1:x2]

    def get_instance_annotation(self, ins_anno_data: dict, image_dict: dict) -> dict:
        """
        Rearranges the keypoints with respect to the instance bounding box
        :param ins_anno_data: keypoints annotation data of the image
        :param image_dict: dictionary containing image metadata information
        :return: extracted instance keypoint annotation data
        """
        ins_bbox = self.get_bbox_data(ins_anno_data, image_dict)
        ins_keypoints = np.array(ins_anno_data['keypoints']).reshape(-1, 3)
        ins_keypoints = ins_keypoints - [ins_bbox[0], ins_bbox[1], 0]
        ins_keypoints[ins_keypoints < 0] = 0

        dict_data = {'bbox': ins_bbox,
                     'keypoints': ins_keypoints.flatten().tolist()}
        return dict_data

    def get_bbox_data(self, ins_anno_data: dict, image_dict: dict) -> list:
        """
        Get bounding box from the instance annotation dictionary
        :param ins_anno_data: input instance annotation dictionary
        :param image_dict: image metadata containing image shape information
        :return: bounding box
        """
        if self.expand_bbox:
            img_height = image_dict['height']
            img_width = image_dict['width']
            bbox_x, bbox_y, bbox_w, bbox_h = ins_anno_data['bbox']
            x1 = max(0, int(bbox_x - bbox_w * self.expand_bbox / 2.0))
            y1 = max(0, int(bbox_y - bbox_h * self.expand_bbox / 2.0))
            x2 = min(img_width, int(x1 + bbox_w + (bbox_w * self.expand_bbox)))
            y2 = min(img_height, int(y1 + bbox_h + (bbox_h * self.expand_bbox)))
            ins_bbox = [x1, y1, x2 - x1, y2 - y1]
        else:
            ins_bbox = [int(value) for value in ins_anno_data['bbox']]
        return ins_bbox

    def get_metadata_dict(self, image_metadata_dict: dict, image_shape: list, filename: str,
                          anno_data: dict) -> dict:
        """
        Creates a dictionary with metadata information
        :param image_metadata_dict: coco image metadata
        :param image_shape: image shape as heightxwidth
        :param filename: instance image filename
        :param anno_data: instance annotation data
        :return: dictionary containing metadata of an instance
        """
        data_dict = {'file_name': filename,
                     'original_file_name': image_metadata_dict['file_name'],
                     'width': image_shape[1],
                     'height': image_shape[0]}

        if 'num_keypoints' in anno_data:
            data_dict['num_keypoints'] = anno_data['num_keypoints']
        else:
            data_dict['num_keypoints'] = int(0)

        if 'area' in anno_data:
            data_dict['area'] = anno_data['area']
        else:
            data_dict['area'] = float(0.0)

        if 'id' in anno_data:
            data_dict['org_anno_id'] = str(anno_data['id'])
        else:
            data_dict['org_anno_id'] = ''

        if 'license' in image_metadata_dict:
            data_dict['license'] = image_metadata_dict['license']
        return data_dict

    def get_bodyparts_data(self, anno_data: dict) -> dict:
        """
        Creates dictionary for keypoints of different body parts
        :param anno_data: instance keypoint annotation data
        :return: dictionary containing keypoints grouped into body parts
        """
        ins_bodyparts_dict = {}
        ins_keypoints = np.array(anno_data['keypoints']).reshape(len(self.person_keypoints), -1)
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
        return ins_bodyparts_dict

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

    def __len__(self):
        """
        :return: Returns length of the dataset.
        """
        return len(self.all_image_ids)


def argument_parse():
    """
    Argument parser
    :return: parsed arguments
    """
    default_coco_dir_path = Path(f'D:\DataSets\coco_data')
    default_split = 'val'
    default_expand_bbox = 0.0
    default_filter_on_min_area = 0.0
    default_filter_on_min_keypoints = 0
    default_always_crop = True

    parser = argparse.ArgumentParser(
        description="Transform COCO instance keypoint annotations to the custom json format")
    parser.add_argument("--path", type=str, default=default_coco_dir_path,
                        help="Path to the main coco directory")
    parser.add_argument("--split", type=str, default=default_split,
                        help="Coco split 'train' or 'val' to be used")
    parser.add_argument("--expand_bbox", type=float, default=default_expand_bbox,
                        help="Factor to expand the instance bounding box")
    parser.add_argument("--filter_on_min_area", type=float, default=default_filter_on_min_area,
                        help="Minimum area below which instance is ignored")
    parser.add_argument("--filter_on_min_keypoints", type=int,
                        default=default_filter_on_min_keypoints,
                        help="Minimum number of keypoints below which instance is ignored")
    parser.add_argument('--always_crop', action='store_true', default=default_always_crop,
                        help="Crops the image with bounding box even if there is only one "
                             "instance in the image")
    args = parser.parse_args()
    print('Arguments:')
    print(vars(args))
    return args


def main():
    args = argument_parse()

    coco_dataset = CocoInstanceDataset(Path(args.path), args.split,
                                       expand_bbox=args.expand_bbox,
                                       filter_on_min_area=args.filter_on_min_area,
                                       filter_on_min_keypoints=args.filter_on_min_keypoints,
                                       always_crop=args.always_crop)
    # coco_dataset = CocoInstanceDataset(coco_dir_path, split)

    print('Execution started!')
    start_time = time.time()

    json_data = coco_dataset.get_annotation_in_json()
    with open(Path(args.path) / f'coco_instance_keypoint_{args.split}2017.json', 'w') as f:
        json.dump(json_data, f)

    stop_time = time.time()
    print('Execution completed!')
    print(f'Time of execution: {np.round(stop_time - start_time, 2)} Sec')
    print(f'Time of execution: {stop_time - start_time}')
    print(f'Number of generated instances: {len(json_data)}')


if __name__ == '__main__':
    main()

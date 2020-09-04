from easydict import EasyDict
import cv2

from external.pytorch_YOLOv4.tool.utils import *
from external.pytorch_YOLOv4.tool.torch_utils import *
from external.pytorch_YOLOv4.tool.yolo_layer import YoloLayer
from external.pytorch_YOLOv4.tool.utils import load_class_names, plot_boxes_cv2
from external.pytorch_YOLOv4.tool.torch_utils import do_detect
from external.pytorch_YOLOv4.models import *


class CarDetector:
    def __init__(self, config):
        r"""YOLOv4 car detector

        Args:
            config (EasyDict): Configuration for YOLOv4 detector
                .n_classes (int): Number of classes for detection. Default 80
                .weights_path (int): path to .pth file containing model weights
                .device (str): Device to run the detector

                .class_name_file (str): Path to class name file

                .image_size (int): Input size of model, i.e. (image_size, image_size)
                .conf_thresh (float): Threshold for bbox confidence score
                .nms_thresh (float): Threshold for NMS
                .size_threshold (tuple(int)): Tuple of two integers, i.e. (w, h)
                    Any bounding box, whose width is less than w, or height is less
                    than h, is discarded
        """
        self.n_classes = config.n_classes
        self.image_size = config.image_size
        self.conf_thresh = config.conf_thresh 
        self.nms_thresh = config.nms_thresh
        self.size_threshold = size_threshold
        self.device = config.device

        self.model = self.__load_model(
            config.n_classes, config.weights_path
        )
        self.idx2class = self.__load_class_names(config.class_name_file)
    
    def __load_model(self, n_classes, weights_path):
        model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True).to(self.device)
        model = self.__load_weights(model, weights_path)
        model = model.eval()
        return model
    
    def __load_weights(self, model, weights_path):
        state_dict = torch.load(weights_path, map_location=torch.device(self.device))
        model.load_state_dict(state_dict)
        return model
    
    def __load_class_names(self, path):
        with open(path, 'r') as fp:
            idx2class = [line.rstrip() for line in fp.readlines()]
        return idx2class
    
    def detect(self, image):
        r"""Run car detection to return a list of bounding boxes

        Args:
            image (np.array): Input image. This is a np.ndarray of shape (h, w, 3)
        Returns:
            list: A Python list of shape (n_bboxes, 7). Each bounding box is of 
                the form (x1, y1, x2, y2, class_confidence, class_confidence, class_id) 
                where (x1, y1, x2, y2) are relative coordinates of the bounding box
        """
        image = self.__preprocess(image)
        preds = self.model(image)
        preds = self.__postprocess(image, preds)
        return preds
    
    def __preprocess(self, image):
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()
        image = image[None, ...] / 255.0
        return image.to(self.device)
    
    def __postprocess(self, image, preds):
        boxes = self.__parse_preds(image, preds)
        boxes = self.__filter_bboxes(image, boxes)
        return boxes
    
    def __parse_preds(self, image, preds):
        box_array = preds[0].cpu().detach().numpy()
        box_array = box_array[:, :, 0]
        
        confs = preds[1].cpu().detach().numpy()
        max_conf = np.max(confs, axis=2)
        max_id = np.argmax(confs, axis=2)

        bboxes_batch = []
        for i in range(box_array.shape[0]):
            argwhere = max_conf[i] > self.conf_thresh
            l_box_array = box_array[i, argwhere, :]
            l_max_conf = max_conf[i, argwhere]
            l_max_id = max_id[i, argwhere]

            bboxes = []
            for j in range(self.n_classes):

                cls_argwhere = l_max_id == j
                ll_box_array = l_box_array[cls_argwhere, :]
                ll_max_conf = l_max_conf[cls_argwhere]
                ll_max_id = l_max_id[cls_argwhere]

                keep = nms_cpu(ll_box_array, ll_max_conf, self.nms_thresh)
                
                if (keep.size > 0):
                    ll_box_array = ll_box_array[keep, :]
                    ll_max_conf = ll_max_conf[keep]
                    ll_max_id = ll_max_id[keep]

                    for k in range(ll_box_array.shape[0]):
                        bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])
            
            bboxes_batch.append(bboxes)
        return bboxes_batch[0]

    def __filter_bboxes(self, image, boxes):
        _, _, h, w = image.shape
        boxes = np.array(boxes)
        hbox = (boxes[:, 3] - boxes[:, 1]) * h
        wbox = (boxes[:, 2] - boxes[:, 0]) * w
        boxes = boxes[np.logical_and(
            hbox >= self.size_threshold[1],
            wbox >= self.size_threshold[0]
        )]
        return boxes


if __name__ == "__main__":
    config = EasyDict(dict(
        n_classes=80,
        weights_path="/content/drive/My Drive/CarDetection/weights/car_detection/yolov4.pth",
        class_name_file="/content/drive/My Drive/CarDetection/pytorch_YOLOv4/data/coco.names",
        device="cuda:0",
        image_size=416,
        conf_thresh=0.4,
        nms_thresh=0.6
    ))
    detector = CarDetector(config)

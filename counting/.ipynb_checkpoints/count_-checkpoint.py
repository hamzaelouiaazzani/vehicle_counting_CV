from ultralytics.utils.plotting import Annotator, colors
from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import deprecation_warn

import torch
import numpy as np
import cv2

import sys
from copy import deepcopy
import json
import os
from pathlib import Path

class args:
    """
    This class contains configuration parameters for the vehicle counting system using the YOLO model and various tracking approaches.

    Attributes:
        source (str): Filename of the video to perform counting on.
                      Need to be set.
        name (str): Name of the folder for the current experiment results.
                    Need to be set.
        yolo_model (Path): Path to the YOLO model file.
                           Default is 'yolov8n.pt'.
        reid_model (Path): Path to the re-identification model file used if the tracker employs appearance description of objects.
                           Examples include 'osnet_x0_25_market1501.pt', 'mobilenetv2_x1_4_msmt17.engine', etc.
        tracking_method (str): Method used for tracking. Options include 'bytetrack', 'botsort', 'strongsort', 'ocsort', 'deepocsort', and 'hybridsort'.
        imgsz (list): Input size of the frames.
                      Default is [640].
        conf (float): Confidence threshold for detection.
                      Default is 0.6.
        iou (float): Intersection over Union (IoU) threshold.
                     Default is 0.7.
        device (str): Device used for running the model (GPU by default).
                      Default is ''.
        show (bool): Whether to display the video scene. Not supported in Google Colab.
                     Default is False.
        save (bool): Whether to save the videos illustrating the tracking results.
                     Default is True.
        classes (list): List of class indices to detect.
                        Default is [1, 2, 3, 5, 7] (vehicles).
        project (str): Folder to save the tracking results.
                       Default is 'runs/count'.
        exist_ok (bool): Whether to overwrite existing results.
                         Default is True.
        half (bool): Whether to use half-precision (16-bit floating-point format) to reduce memory consumption.
                     Default is False.
        vid_stride (int): Frame stride, e.g., process all frames with stride=1 or process every other frame with stride=2.
                          Default is 1.
        show_labels (bool): Whether to display labels (e.g., car, truck, bus) in the saved video results.
                            Default is True.
        show_conf (bool): Whether to display confidence scores of detections.
                          Default is False.
        save_txt (bool): Whether to save results in a text file format.
                         Default is False.
        save_id_crops (bool): Whether to save tracking results for each object in frames.
                              Default is True.
        save_mot (bool): Whether to save tracking results in a report file.
                         Default is True.
        line_width (int): Line width of the bounding boxes.
                          Default is None.
        per_class (bool): Whether to count per class.
                          Default is True.
        verbose (bool): Whether to enable verbose logging.
                        Default is False.
        counting_approach (str): Approach for counting vehicles. Options include 'detection_only', 'tracking_without_line', 'tracking_with_line', 'tracking_with_two_lines'.
                                 Default is 'tracking_with_two_lines'.
        line_point11 (tuple): Coordinates of the first point of the first line. Values between 0 and 1 indicate percentages.
                              For example, (0.4, 0.0) means 40% of the frame width (pixel 0.4 * image width) and 0% of the frame height (pixel 0).
                              When masking the video frames with included_box, it becomes 0.4 * new width after mask.
        line_point12 (tuple): Coordinates of the second point of the first line. Values between 0 and 1 indicate percentages.
                              For example, (0.3, 1.0) means 30% of the frame width (pixel 0.3 * image width) and 100% of the frame height (pixel image height).
        line_vicinity (float): Vicinity of the line for counting. This argument is used in the 'detection_only' or 'tracking_with_line' counting approaches and ignored otherwise ('tracking_without_line' or 'tracking_with_two_lines').
                               Default is 0.1.
        line_point21 (tuple): Coordinates of the first point of the second line. Values between 0 and 1 indicate percentages.
                              For example, (0.6, 0.0) means 60% of the frame width (pixel 0.6 * image width) and 0% of the frame height (pixel 0).
                              This argument is considered only in the 'tracking_with_two_lines' counting approach and ignored otherwise.
        line_point22 (tuple): Coordinates of the second point of the second line. Values between 0 and 1 indicate percentages.
                              For example, (0.7, 1.0) means 70% of the frame width (pixel 0.7 * image width) and 100% of the frame height (pixel image height).
                              This argument is considered only in the 'tracking_with_two_lines' counting approach and ignored otherwise.
        use_mask (bool): Whether to use a mask for preprocessing. If set to False, 'visualize_masked_frames' and 'included_box' arguments will be ignored.
                         If set to True, the percentages for 'line_point11', 'line_point12', 'line_point21', and 'line_point22' will be transformed to pixel values with respect to the included_box.
                         Default is False.
        visualize_masked_frames (bool): Whether to visualize masked frames.
                                        Default is True.
        included_box (list): Box coordinates for masking, specified as percentages between -1 and 1. For example, [0.1, 0.2, -0.2, -0.1] indicates:
                             - The first two values (0.1, 0.2) represent the TOP-LEFT point of the included rectangle when using a mask for frames. 
                               This point is 10% of the width and 20% of the height.
                             - The last two values (-0.2, -0.1) represent the BOTTOM-RIGHT point of the included rectangle after masking. 
                               This point is 80% of the width and 90% of the height.
    """
    source = "kech1.mp4"
    name = "kech1"
    yolo_model = Path('yolov8n.pt')
    reid_model = Path("osnet_x0_25_msmt17.pt")
    tracking_method = "ocsort"
    imgsz = [640]
    conf = 0.6
    iou = 0.7
    device = ''
    show = False
    save = True
    classes = [1, 2, 3, 5, 7]
    project = "runs/count"
    exist_ok = True
    half = False
    vid_stride = 1
    show_labels = True
    show_conf = False
    save_txt = False
    save_id_crops = True
    save_mot = True
    line_width = None
    per_class = True
    verbose = False
    counting_approach = "tracking_with_two_lines"
    line_point11 = (0.4, 0.0)
    line_point12 = (0.3, 1.0)
    line_vicinity = 0.01
    line_point21 = (0.6, 0.0)
    line_point22 = (0.7, 1.0)
    use_mask = False
    visualize_masked_frames = True
    included_box = [0.1, 0.2, -0.2, -0.1]



class Annotator_for_counting(Annotator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def draw_line_with_two_points(self, line_point1, line_point2, color=(255, 255, 255), width=None):
        """
        Draws a line given two points on the image.

        Args:
            line_point1 (tuple): Coordinates of the first point (x1, y1).
            line_point2 (tuple): Coordinates of the second point (x2, y2).
            color (tuple, optional): RGB color tuple for the line. Default is white (255, 255, 255).
            width (int, optional): Width of the line. Default is the line width specified during initialization.

        Note:
            This method modifies the image in place.
        """
        width = width or self.lw

        if self.pil:
            self.draw.line([line_point1, line_point2], fill=color, width=width)
        else:
            cv2.line(self.im, line_point1, line_point2, color, thickness=width, lineType=cv2.LINE_AA)


class counter_YOLO(YOLO):
    def __init__(self, args):
        """
        Initializes the counter_YOLO object and sets up counting attributes and video information.

        Args:
            args: Arguments containing configurations for the YOLO model and counting approach.
        """
        super().__init__(args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt')

        self.counting_attributes = {}
        self.counting_preprocess = {}
        self.video_attributes = {}

        self.get_video_info(args.source)

        with open("counting/index_to_labels.json", "r") as json_file:
            index_to_labels = json.load(json_file)
        self.counting_attributes["index_to_label"] = index_to_labels  
        self.counting_attributes["line_vicinity"] = None
        self.counting_attributes["intercept_line1"] = None
        self.counting_attributes["slope1"] = None
        self.counting_attributes["intercept_line2"] = None
        self.counting_attributes["slope2"] = None
    
        self.counting_attributes["line_point21"], self.counting_attributes["line_point22"] = args.line_point21, args.line_point22
        self.right_of_line = {}
        
        self.max_cls_index = max(list(args.classes))+1 if args.classes is not None else 80
        self.counter = 0
        self.count_per_class = 0
        self.ids_set = set()
        self.ids_filtered = torch.tensor([])
        self.ids_frames = torch.tensor([])
        self.id_to_first_last = {}
        

        try:
            if args.counting_approach == "detection_only":
                print("You are following the detection only in the vicinity of a given line approach.")

                self.counting_attributes["counting_approach"] = "detection_only"
                self.counting_attributes["with_track"], self.counting_attributes["with_line"] = False, True
                self.counting_attributes["line_point11"], self.counting_attributes["line_point12"] = args.line_point11, args.line_point12

                self.counting_attributes["line_vicinity"] = args.line_vicinity

                self.counting_preprocess["use_mask"] = args.use_mask
                if args.use_mask:
                    self.counting_preprocess["visualize_masked_frames"] = args.visualize_masked_frames
                    self.preprocess_included_box_mask(args)
                    self.slope_intercept_with_mask("1")
                else:
                    self.slope_intercept_without_mask("1")
                
                self.set_counting_function(self.count_detect_line)
                self.pipeline_function = self.pipeline_without_tracking

            elif args.counting_approach == "tracking_without_line":
                print("You are following the detection&tracking over the whole frame spatial information approach.")

                self.counting_attributes["counting_approach"] = "tracking_without_line"
                self.counting_attributes["with_track"], self.counting_attributes["with_line"] = True, False
                self.counting_attributes["line_point11"], self.counting_attributes["line_point12"] = None, None

                self.counting_preprocess["use_mask"] = args.use_mask
                if args.use_mask:
                    self.counting_preprocess["visualize_masked_frames"] = args.visualize_masked_frames
                    self.preprocess_included_box_mask(args)

                self.set_counting_function(self.count_track_noline)
                self.pipeline_function = self.pipeline_with_tracking

            elif args.counting_approach == "tracking_with_line":
                print("You are following the detection&tracking in the vicinity of pre-defined line approach.")

                self.counting_attributes["counting_approach"] = args.counting_approach
                self.counting_attributes["with_track"], self.counting_attributes["with_line"] = True, True
                self.counting_attributes["line_point11"], self.counting_attributes["line_point12"] = args.line_point11, args.line_point12
                
                self.counting_attributes["line_vicinity"] = args.line_vicinity

                self.counting_preprocess["use_mask"] = args.use_mask
                if args.use_mask:
                    self.counting_preprocess["visualize_masked_frames"] = args.visualize_masked_frames
                    self.preprocess_included_box_mask(args)
                    self.slope_intercept_with_mask("1")
                else:
                    self.slope_intercept_without_mask("1")
                
                self.pipeline_function = self.pipeline_with_tracking
                self.set_counting_function(self.count_track_line)   
                
            elif args.counting_approach == "tracking_with_two_lines":
                print("You are following the detection&tracking with two lines approach.")
                
                self.counting_attributes["counting_approach"] = "tracking_with_two_lines"
                self.counting_attributes["with_track"], self.counting_attributes["with_line"] = True, True
                self.counting_attributes["line_point11"], self.counting_attributes["line_point12"] = args.line_point11, args.line_point12
                self.counting_attributes["line_point21"], self.counting_attributes["line_point22"] = args.line_point21, args.line_point22

                self.counting_attributes["line_vicinity"] = None
               
                self.counting_preprocess["use_mask"] = args.use_mask
                if args.use_mask:
                    self.counting_preprocess["visualize_masked_frames"] = args.visualize_masked_frames
                    self.preprocess_included_box_mask(args)
                    self.slope_intercept_with_mask("1")
                    self.slope_intercept_with_mask("2")
                else:
                    self.slope_intercept_without_mask("1")
                    self.slope_intercept_without_mask("2")

                for i in range(2):
                    if self.counting_attributes["line_point"+str(i+1)+"1"][0] == self.counting_attributes["line_point"+str(i+1)+"2"][0]:
                        self.set_right_of_line(self.is_right_vertical, str(i+1))
                    elif self.counting_attributes["slope"+str(i+1)] > 0:
                        self.set_right_of_line(self.is_right_positive_slope, str(i+1))
                    else:
                        self.set_right_of_line(self.is_right_negative_zero_slope, str(i+1))
                        
                self.pipeline_function = self.pipeline_with_tracking
                self.set_counting_function(self.count_track_two_lines)
                
            else:
                raise ValueError("Please make sure you have chosen one of the three available counting approaches via one of the following strings: detection_only_approach, tracking_without_line, or tracking_with_line")
                
        except ValueError as e:
            print("Error:", e)
            print("Please ensure the arguments are correctly specified according to the provided information.")
            sys.exit(1)

    def preprocess_included_box_mask(self, args):
        """
        Preprocesses the included box mask for counting.

        Args:
            args: Arguments containing the included box coordinates.
        """
        x_ib1, y_ib1 = int(args.included_box[0] * self.video_attributes["width"]), int(args.included_box[1] * self.video_attributes["height"])
        x_ib2, y_ib2 = int(args.included_box[2] * self.video_attributes["width"]), int(args.included_box[3] * self.video_attributes["height"])
        self.counting_preprocess["included_box"] = [x_ib1, y_ib1, x_ib2, y_ib2]
        
        for i in range(len(self.counting_preprocess["included_box"])):
            if self.counting_preprocess["included_box"][i] < 0:
                if i == 0 or i == 2:
                    self.counting_preprocess["included_box"][i] = int(self.video_attributes["width"] + self.counting_preprocess["included_box"][i])
                if i == 1 or i == 3:
                    self.counting_preprocess["included_box"][i] = int(self.video_attributes["height"] + self.counting_preprocess["included_box"][i])

    def slope_intercept_with_mask(self, line_num):
        """
        Calculates the slope and intercept for a line with a mask.

        Args:
            line_num (str): Line number ("1" or "2").
        """
        x1, y1 = self.counting_attributes["line_point"+str(line_num)+"1"]
        x2, y2 = self.counting_attributes["line_point"+str(line_num)+"2"]

        x_ib1, y_ib1 = self.counting_preprocess["included_box"][0], self.counting_preprocess["included_box"][1]
        x_ib2, y_ib2 = self.counting_preprocess["included_box"][2], self.counting_preprocess["included_box"][3]
        x1, y1 = int(x_ib1 + x1 * (x_ib2 - x_ib1)), int(y_ib1 + y1 * (y_ib2 - y_ib1))
        x2, y2 = int(x_ib1 + x2 * (x_ib2 - x_ib1)), int(y_ib1 + y2 * (y_ib2 - y_ib1))
        self.counting_attributes["line_point"+str(line_num)+"1"] = (x1, y1)
        self.counting_attributes["line_point"+str(line_num)+"2"] = (x2, y2)

        if x1 == x2:
            self.counting_attributes["slope"+str(line_num)] = "inf"
            self.counting_attributes["intercept_line"+str(line_num)] = x1
        else:
            self.counting_attributes["slope"+str(line_num)] = (y2 - y1) / (x2 - x1)
            self.counting_attributes["intercept_line"+str(line_num)] = y1 - self.counting_attributes["slope"+str(line_num)] * x1

        if self.counting_attributes["slope1"] == "inf":
            self.set_distance_function(self.dist_v_bbox_line)
        elif self.counting_attributes["slope1"] == 0:
            self.set_distance_function(self.dist_h_bbox_line)
        elif self.counting_attributes["slope1"]:
            self.set_distance_function(self.dist_s_bbox_line)

    def slope_intercept_without_mask(self, line_num):
        """
        Calculates the slope and intercept for a line without a mask.

        Args:
            line_num (str): Line number ("1" or "2").
        """
        x1, y1 = self.counting_attributes["line_point"+str(line_num)+"1"]
        x2, y2 = self.counting_attributes["line_point"+str(line_num)+"2"]

        x1, y1 = int(x1 * self.video_attributes["width"]), int(y1 * self.video_attributes["height"])
        x2, y2 = int(x2 * self.video_attributes["width"]), int(y2 * self.video_attributes["height"])
        self.counting_attributes["line_point"+str(line_num)+"1"], self.counting_attributes["line_point"+str(line_num)+"2"] = (x1, y1), (x2, y2)

        if x1 == x2:
            self.counting_attributes["slope"+str(line_num)] = "inf"
            self.counting_attributes["intercept_line"+str(line_num)] = x1
        else:
            self.counting_attributes["slope"+str(line_num)] = (y2 - y1) / (x2 - x1)
            self.counting_attributes["intercept_line"+str(line_num)] = y1 - self.counting_attributes["slope"+str(line_num)] * x1

        if self.counting_attributes["slope1"] == "inf":
            self.set_distance_function(self.dist_v_bbox_line)
        elif self.counting_attributes["slope1"] == 0:
            self.set_distance_function(self.dist_h_bbox_line)
        elif self.counting_attributes["slope1"]:
            self.set_distance_function(self.dist_s_bbox_line)

    def get_video_info(self, source):
        """
        Retrieves video properties (width, height, frame rate, total frames) from the video source.

        Args:
            source: Path to the video file.
        """
        video_path = os.path.join(os.getcwd(), source)
        video_capture = cv2.VideoCapture(video_path)

        self.video_attributes["width"] = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_attributes["height"] = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_attributes["frame_rate"] = int(video_capture.get(cv2.CAP_PROP_FPS))
        self.video_attributes["total_frames"] = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        video_capture.release()

    def preprocess_for_counting(self, im0s):
        """
        Preprocesses the images for counting by applying masks if necessary.

        Args:
            im0s: List of images.

        Returns:
            Preprocessed images and original images.
        """
        def mask_list_image(im0s, included_box):
            masked_images = []
            for img in im0s:
                x1, y1, x2, y2 = self.counting_preprocess["included_box"]
                mask = np.ones_like(img)
                mask[:y1, :, :] = 0
                mask[y2:, :, :] = 0
                mask[:, :x1, :] = 0
                mask[:, x2:, :] = 0
                masked_img = img * mask
                masked_images.append(masked_img)
            return masked_images

        if self.counting_preprocess["use_mask"]:
            if self.counting_preprocess["visualize_masked_frames"]:
                im0s = mask_list_image(im0s, self.counting_preprocess["included_box"])
                im = self.predictor.preprocess(im0s)
            else:
                im0s_ = mask_list_image(im0s, self.counting_preprocess["included_box"])
                im = self.predictor.preprocess(im0s_)
        else:
            im = self.predictor.preprocess(im0s)

        return im, im0s

    def postprocess_for_counting(self, preds, im, im0s, path):
        """
        Post-processes the predictions to obtain results for counting.

        Args:
            preds: Predictions from the model.
            im: Preprocessed images.
            im0s: Original images.
            path: Path to the video frame.
        """
        if isinstance(self.predictor.model, AutoBackend):
            self.predictor.results = self.predictor.postprocess(preds, im, im0s)
        else:
            self.predictor.results = self.predictor.model.postprocess(path, preds, im, im0s)

    def dist_h_bbox_line(self, bbox, intercept, slope=0):
        """
        Calculates the horizontal distance from bounding boxes to the line.

        Args:
            bbox: Tensor of shape (m, 2), where m is the number of bounding boxes in a single frame.
            intercept: Intercept of the line.
            slope: Slope of the line (default is 0).

        Returns:
            Horizontal distance from bounding boxes to the line.
        """
        y = bbox[:, 1]
        return abs(y - intercept)

    def dist_v_bbox_line(self, bbox, intercept, slope=None):
        """
        Calculates the vertical distance from bounding boxes to the line.

        Args:
            bbox: Tensor of shape (m, 2), where m is the number of bounding boxes in a single frame.
            intercept: Intercept of the line.
            slope: Slope of the line (default is None).

        Returns:
            Vertical distance from bounding boxes to the line.
        """
        x = bbox[:, 0]
        return abs(x - intercept)

    def dist_s_bbox_line(self, bbox, intercept, slope):
        """
        Calculates the distance from bounding boxes to the line with a given slope.

        Args:
            bbox: Tensor of shape (m, 2), where m is the number of bounding boxes in a single frame.
            intercept: Intercept of the line.
            slope: Slope of the line.

        Returns:
            Distance from bounding boxes to the line.
        """
        x, y = bbox[:, 0], bbox[:, 1]
        A = slope
        B = -1
        C = intercept        
        numerator = abs(A * x + B * y + C)
        denominator = np.sqrt(A**2 + B**2)
        return numerator / denominator

    def set_distance_function(self, distance_function):
        """
        Sets the distance function to be used for counting.

        Args:
            distance_function: Distance function to set.
        """
        self.distance_function = distance_function

    def is_right_vertical(self, bbox, intercept, slope="inf"):
        """
        Determines if bounding boxes are to the right of a vertical line.

        Args:
            bbox: Tensor of shape (m, 2), where m is the number of bounding boxes in a single frame.
            intercept: Intercept of the line.
            slope: Slope of the line (default is "inf").

        Returns:
            Boolean array indicating if bounding boxes are to the right of the vertical line.
        """
        return bbox[:, 0] > intercept

    def is_right_positive_slope(self, bbox, intercept, slope):
        """
        Determines if bounding boxes are to the right of a line with a positive slope.

        Args:
            bbox: Tensor of shape (m, 2), where m is the number of bounding boxes in a single frame.
            intercept: Intercept of the line.
            slope: Slope of the line.

        Returns:
            Boolean array indicating if bounding boxes are to the right of the line with a positive slope.
        """
        return bbox[:, 1] < slope * bbox[:, 0] + intercept

    def is_right_negative_zero_slope(self, bbox, intercept, slope):
        """
        Determines if bounding boxes are to the right of a line with a negative or zero slope.

        Args:
            bbox: Tensor of shape (m, 2), where m is the number of bounding boxes in a single frame.
            intercept: Intercept of the line.
            slope: Slope of the line.

        Returns:
            Boolean array indicating if bounding boxes are to the right of the line with a negative or zero slope.
        """
        return bbox[:, 1] > slope * bbox[:, 0] + intercept

    def set_right_of_line(self, right_of_line, line_num):
        """
        Sets the function to determine if bounding boxes are to the right of the line.

        Args:
            right_of_line: Function to determine if bounding boxes are to the right of the line.
            line_num (str): Line number ("1" or "2").
        """
        self.right_of_line["line"+str(line_num)] = right_of_line

    def count_detect_line(self, boxes, intercept1, slope1, line_vicinity, intercept2=None, slope2=None):
        """
        Counts vehicles based on detection only in the vicinity of a line.

        Args:
            boxes: Detected bounding boxes.
            intercept1: Intercept of the first line.
            slope1: Slope of the first line.
            line_vicinity: Vicinity of the line.
            intercept2: Intercept of the second line (optional).
            slope2: Slope of the second line (optional).
        """
        bboxs = boxes.xywh[:, :2]
        
        if bboxs.numel() > 0:
            dist = self.distance_function(bbox=bboxs, intercept=intercept1, slope=slope1) < line_vicinity * torch.max(boxes.xywh[:, 2], boxes.xywh[:, 3])
            cls = boxes.cls.cpu().int()
            cls = torch.eye(self.max_cls_index)[cls]
            mask = ~dist[:, None].expand(-1, self.max_cls_index)
            masked_cls = torch.masked_fill(cls, mask, 0) 
            self.count_per_class += torch.sum(masked_cls, dim=0)
            self.counter += torch.sum(masked_cls)

    def count_track_noline(self, boxes, intercept1=None, slope1=None, line_vicinity=None, intercept2=None, slope2=None):
        """
        Counts vehicles using tracking without considering a line.

        Args:
            boxes: Detected bounding boxes.
            intercept1: Intercept of the first line (optional).
            slope1: Slope of the first line (optional).
            line_vicinity: Vicinity of the line (optional).
            intercept2: Intercept of the second line (optional).
            slope2: Slope of the second line (optional).
        """
        bboxs = boxes.xywh[:, :2]
        ids = boxes.id
        
        if bboxs.numel() > 0 and ids is not None:
            ids_ = torch.tensor([id_ not in self.ids_set for id_ in ids.numpy()], dtype=torch.bool)
            indices = torch.nonzero(ids_)
            ids_to_keep_track = ids[indices]
            
            for id in ids_to_keep_track.view(-1):
                self.ids_set.add(int(id))

            cls = boxes.cls.cpu().int()
            cls = torch.eye(self.max_cls_index)[cls]
            mask = ~ids_[:, None].expand(-1, self.max_cls_index)
            masked_ids = torch.masked_fill(cls, mask, 0)
            self.count_per_class += torch.sum(masked_ids, dim=0)
            self.counter += torch.sum(masked_ids)

    def count_track_line(self, boxes, intercept1, slope1, line_vicinity, intercept2=None, slope2=None):
        """
        Counts vehicles using tracking in the vicinity of a line.

        Args:
            boxes: Detected bounding boxes.
            intercept1: Intercept of the first line.
            slope1: Slope of the first line.
            line_vicinity: Vicinity of the line.
            intercept2: Intercept of the second line (optional).
            slope2: Slope of the second line (optional).
        """
        bboxs = boxes.xywh[:, :2]
        ids = boxes.id
        
        if bboxs.numel() > 0 and ids is not None:
            ids_ = torch.tensor([id_ not in self.ids_set for id_ in ids.numpy()], dtype=torch.bool)
            dist = self.distance_function(bbox=bboxs, intercept=intercept1, slope=slope1) < line_vicinity * torch.max(boxes.xywh[:, 2], boxes.xywh[:, 3])
            indices = torch.nonzero(ids_ & dist)
            ids_to_keep_track = ids[indices]
            for id in ids_to_keep_track.view(-1):
                self.ids_set.add(int(id))
                
            cls = boxes.cls.cpu().int()
            cls = torch.eye(self.max_cls_index)[cls]
            mask = ~ids_[:, None].expand(-1, self.max_cls_index)
            masked_ids = torch.masked_fill(cls, mask, 0)
            mask = ~dist[:, None].expand(-1, self.max_cls_index)
            masked_dist = torch.masked_fill(cls, mask, 0)
            masked_count = torch.logical_and(masked_dist, masked_ids)
            self.count_per_class += torch.sum(masked_count, dim=0)
            self.counter += torch.sum(masked_count)

    def count_track_two_lines(self, boxes, intercept1, slope1, line_vicinity, intercept2, slope2):
        """
        Counts vehicles using tracking with two lines.

        Args:
            boxes: Detected bounding boxes.
            intercept1: Intercept of the first line.
            slope1: Slope of the first line.
            line_vicinity: Vicinity of the line.
            intercept2: Intercept of the second line.
            slope2: Slope of the second line.
        """
        line_vicinity = None
        bboxs = boxes.xywh[:, :2]
        ids = boxes.id
        
        if bboxs.numel() > 0 and ids is not None:
            ids_ = torch.tensor([id_ not in self.ids_set for id_ in ids.numpy()], dtype=torch.bool)
            right_of_line1_vehicles = self.right_of_line["line1"](bboxs, intercept1, slope1)
            left_of_line2_vehicles = ~self.right_of_line["line2"](bboxs, intercept2, slope2)
            vehicles_between_two_lines = right_of_line1_vehicles & left_of_line2_vehicles
            indices = torch.nonzero(ids_ & vehicles_between_two_lines)
            ids_to_keep_track = ids[indices]
            for id in ids_to_keep_track.view(-1):
                self.ids_set.add(int(id))
    
            cls = boxes.cls.cpu().int()
            cls = torch.eye(self.max_cls_index)[cls]
            mask = ~ids_[:, None].expand(-1, self.max_cls_index)
            masked_ids = torch.masked_fill(cls, mask, 0)
            mask = ~vehicles_between_two_lines[:, None].expand(-1, self.max_cls_index)
            masked_between_two_lines = torch.masked_fill(cls, mask, 0)
            masked_count = torch.logical_and(masked_between_two_lines, masked_ids)
            self.count_per_class += torch.sum(masked_count, dim=0)
            self.counter += torch.sum(masked_count)
    
            self.ids_filtered = ids[vehicles_between_two_lines]
            ids_frame = torch.stack((self.ids_filtered, torch.full_like(self.ids_filtered, self.frame_number)), dim=1)
            self.ids_frames = torch.cat((self.ids_frames, ids_frame)).int()
            self.id_to_first_last = self.id_to_first_last_frame(self.ids_frames)    



            
            # Add this block to write to CSV
            import csv 
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with open('vehicle_counts.csv', 'a', newline='') as csvfile:
                fieldnames = ['time', 'ID_between_2_lines', 'class']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write the header only if the file is empty
                if csvfile.tell() == 0:
                    writer.writeheader()
                
                for id, cls_index in zip(self.ids_filtered, cls[ids_to_keep_track]):
                    writer.writerow({'time': current_time, 'ID_between_2_lines': int(id), 'class': int(cls_index.argmax())})


    

    def id_to_first_last_frame(self, ids_frames):
        """
        Gets the first and last frame for each ID in the video.

        Args:
            ids_frames: Tensor of IDs and frame numbers.

        Returns:
            Dictionary mapping each ID to its first and last frame numbers.
        """
        id_to_first_last = {}
        
        for id_, value in ids_frames:
            id_ = int(id_) 
            value = int(value)
            
            if id_ not in id_to_first_last:
                id_to_first_last[id_] = {'first_frame': value, 'last_frame': value}
            else:
                id_to_first_last[id_]['first_frame'] = min(id_to_first_last[id_]['first_frame'], value)
                id_to_first_last[id_]['last_frame'] = max(id_to_first_last[id_]['last_frame'], value)
        
        return id_to_first_last

    def set_counting_function(self, counting_function):
        """
        Sets the counting function to be used.

        Args:
            counting_function: Counting function to set.
        """
        self.counting_function = counting_function

    def run_counting(self, i=0):
        """
        Count the number of vehicles in a video scene (or in a batch of video scenes).

        Args:
            i: The index of the video scene in the batch to consider. If you are only using one scene video, please assign 0 to "i."
        """
        line_vicinity = self.counting_attributes["line_vicinity"]
        intercept1, slope1 = self.counting_attributes["intercept_line1"], self.counting_attributes["slope1"]
        intercept2, slope2 = self.counting_attributes["intercept_line2"], self.counting_attributes["slope2"]
        boxes = self.predictor.results[i].boxes
        self.counting_function(boxes, intercept1, slope1, line_vicinity, intercept2, slope2)

    def pipeline_with_tracking(self, im0s, path, profilers):
        """
        Runs the pipeline with tracking.

        Args:
            im0s: List of images.
            path: Path to the video frame.
            profilers: Profilers for timing each stage of the pipeline.

        Returns:
            Processed images and profiling information.
        """
        with profilers[0]:
            im, im0s = self.preprocess_for_counting(im0s)
        with profilers[1]:
            preds = self.predictor.inference(im)
        with profilers[2]:
            self.postprocess_for_counting(preds, im, im0s, path)
        with profilers[3]:
            self.predictor.run_callbacks('on_predict_postprocess_end')

        return im0s, im, profilers

    def pipeline_without_tracking(self, im0s, path, profilers):
        """
        Runs the pipeline without tracking.

        Args:
            im0s: List of images.
            path: Path to the video frame.
            profilers: Profilers for timing each stage of the pipeline.

        Returns:
            Processed images and profiling information.
        """
        with profilers[0]:
            im, im0s = self.preprocess_for_counting(im0s)
        with profilers[1]:
            preds = self.predictor.inference(im)
        with profilers[2]:
            self.postprocess_for_counting(preds, im, im0s, path)
        with profilers[3]:
            pass

        return im0s, im, profilers

    def set_pipeline_function(self, pipeline_function):
        """
        Sets the pipeline function to be used.

        Args:
            pipeline_function: Pipeline function to set.
        """
        self.pipeline_function = pipeline_function

    def run_pipeline(self, im0s, path, profilers):
        """
        Runs the selected pipeline function.

        Args:
            im0s: List of images.
            path: Path to the video frame.
            profilers: Profilers for timing each stage of the pipeline.

        Returns:
            Processed images and profiling information.
        """
        return self.pipeline_function(im0s, path, profilers)

    def plot(self, idx, counter, count_per_class, line_point11, line_point12, line_point21=None, line_point22=None,
             conf=True, line_width=None, font_size=None, font='Arial.ttf', pil=False, img=None, im_gpu=None, kpt_radius=5, kpt_line=True, labels=True, boxes=True, masks=True, probs=True, **kwargs):
        """
        Plots the results on the image, including bounding boxes, lines, and counters.

        Args:
            idx: Index of the result to plot.
            counter: Total count of vehicles.
            count_per_class: Count of vehicles per class.
            line_point11: First point of the first line.
            line_point12: Second point of the first line.
            line_point21: First point of the second line (optional).
            line_point22: Second point of the second line (optional).
            conf: Whether to show confidence scores (default is True).
            line_width: Width of the lines (default is None).
            font_size: Size of the font for text (default is None).
            font: Font for text (default is 'Arial.ttf').
            pil: Whether to use PIL for plotting (default is False).
            img: Image to plot on (default is None).
            im_gpu: GPU image tensor (default is None).
            kpt_radius: Radius for keypoints (default is 5).
            kpt_line: Whether to plot lines for keypoints (default is True).
            labels: Whether to show labels (default is True).
            boxes: Whether to show bounding boxes (default is True).
            masks: Whether to show masks (default is True).
            probs: Whether to show probabilities (default is True).
            **kwargs: Additional deprecated arguments.
        """
        if img is None and isinstance(self.predictor.results[idx].orig_img, torch.Tensor):
            img = np.ascontiguousarray(self.predictor.results[idx].orig_img[0].permute(1, 2, 0).cpu().detach().numpy()) * 255

        if 'show_conf' in kwargs:
            deprecation_warn('show_conf', 'conf')
            conf = kwargs['show_conf']
            assert type(conf) == bool, '`show_conf` should be of boolean type, i.e, show_conf=True/False'

        if 'line_thickness' in kwargs:
            deprecation_warn('line_thickness', 'line_width')
            line_width = kwargs['line_thickness']
            assert type(line_width) == int, '`line_width` should be of int type, i.e, line_width=3'

        names = self.predictor.results[idx].names
        pred_boxes, show_boxes = self.predictor.results[idx].boxes, boxes
        pred_masks, show_masks = self.predictor.results[idx].masks, masks
        pred_probs, show_probs = self.predictor.results[idx].probs, probs

        annotator = Annotator_for_counting(
            deepcopy(self.predictor.results[idx].orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
            example=names)

        if pred_boxes and show_boxes:
            for d in reversed(pred_boxes):
                c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
                name = ('' if id is None else f'id:{id} ') + names[c]
                label = (f'{name} {conf:.2f}' if conf else name) if labels else None
                annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))

        annotator.text([10, 400], f"counter : {counter}", txt_color=(255, 255, 255))
        annotator.text([10, 200], f"count per class : {count_per_class}", txt_color=(255, 255, 255))

        if pred_probs is not None and show_probs:
            text = ',\n'.join(f'{names[j] if names else j} {pred_probs.data[j]:.2f}' for j in pred_probs.top5)
            x = round(self.predictor.results[idx].orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=(255, 255, 255))

        annotator.draw_line_with_two_points(line_point11, line_point12)
        if self.counting_attributes["counting_approach"] == "tracking_with_two_lines":
            annotator.draw_line_with_two_points(line_point21, line_point22)
        
        return annotator.result()

    def write_results(self, idx, results, batch):
        """
        Writes inference results to a file or directory.

        Args:
            idx: Index of the result to write.
            results: Inference results.
            batch: Batch of images and their paths.

        Returns:
            Log string containing information about the written results.
        """
        p, im, _ = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]
        if self.predictor.source_type.webcam or self.predictor.source_type.from_img or self.predictor.source_type.tensor:
            log_string += f'{idx}: '
            frame = self.predictor.dataset.count
        else:
            frame = getattr(self.predictor.dataset, 'frame', 0)
        self.predictor.data_path = p
        self.predictor.txt_path = str(self.predictor.save_dir / 'labels' / p.stem) + ('' if self.predictor.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]
        result = results[idx]

        log_string += result.verbose()

        if self.predictor.args.save or self.predictor.args.show:
            plot_args = {
                'idx': idx,
                'counter': self.counter,
                'count_per_class': self.count_per_class,
                'line_point11': self.counting_attributes["line_point11"],
                'line_point12': self.counting_attributes["line_point12"],
                'line_point21': self.counting_attributes["line_point21"],
                'line_point22': self.counting_attributes["line_point22"],
                'line_width': self.predictor.args.line_width,
                'boxes': self.predictor.args.show_boxes,
                'conf': self.predictor.args.show_conf,
                'labels': self.predictor.args.show_labels}

            if not self.predictor.args.retina_masks:
                plot_args['im_gpu'] = im[idx]

            self.predictor.plotted_img = self.plot(**plot_args)

        if self.predictor.args.save_txt:
            result.save_txt(f'{self.predictor.txt_path}.txt', save_conf=self.predictor.args.save_conf)
        if self.predictor.args.save_crop:
            result.save_crop(save_dir=self.predictor.save_dir / 'crops',
                              file_name=self.predictor.data_path.stem + ('' if self.predictor.dataset.mode == 'image' else f'_{frame}'))

        return log_string

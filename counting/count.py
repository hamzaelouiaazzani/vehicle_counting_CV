from ultralytics.utils.plotting import Annotator , colors
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




class Annotator_for_counting(Annotator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def draw_line_with_two_points(self, line_point1, line_point2, color=(255, 255, 255), width=None):
        """
        Draws a line given two points on the image.

        Args:
            line_point11 (tuple): Coordinates of the first point (x1, y1).
            line_point12 (tuple): Coordinates of the second point (x2, y2).
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
    
        self.counting_attributes["line_point21"] , self.counting_attributes["line_point22"] = args.line_point21 , args.line_point22
        self.right_of_line = {}
        
        self.max_cls_index = max(list(args.classes))+1 if args.classes is not None else 80
        self.counter = 0
        self.count_per_class = 0
        self.ids_frames = torch.tensor([])

        self.count_record = []
        self.ids_set = set()

        try:

            if args.counting_approach == "detection_only_approach":
                print("You are following the detection only in the vicinity of a given line approach.")

                self.counting_attributes["counting_approach"] = "detection_only_approach"
                self.counting_attributes["with_track"] , self.counting_attributes["with_line"] = False , True
                self.counting_attributes["line_point11"] , self.counting_attributes["line_point12"] = args.line_point11 , args.line_point12

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
                self.counting_attributes["with_track"] , self.counting_attributes["with_line"] = True , False
                self.counting_attributes["line_point11"] , self.counting_attributes["line_point12"] = None , None

                self.counting_preprocess["use_mask"] = args.use_mask
                if args.use_mask:
                    self.counting_preprocess["visualize_masked_frames"] = args.visualize_masked_frames
                    self.preprocess_included_box_mask(args)

                self.set_counting_function(self.count_track_noline)
                self.pipeline_function = self.pipeline_with_tracking

            elif args.counting_approach == "tracking_with_line" :
                print("You are following the detection&tracking in the vicinity of pre-defined line approach.")

                self.counting_attributes["counting_approach"] = args.counting_approach
                self.counting_attributes["with_track"] , self.counting_attributes["with_line"] = True , True
                self.counting_attributes["line_point11"] , self.counting_attributes["line_point12"] = args.line_point11 , args.line_point12
                
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
                self.counting_attributes["with_track"] , self.counting_attributes["with_line"] = True , True
                self.counting_attributes["line_point11"] , self.counting_attributes["line_point12"] = args.line_point11 , args.line_point12
                self.counting_attributes["line_point21"] , self.counting_attributes["line_point22"] = args.line_point21 , args.line_point22

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
                        self.set_right_of_line(self.is_right_vertical , str(i+1))
                    elif self.counting_attributes["slope"+str(i+1)] > 0:
                        self.set_right_of_line(self.is_right_positive_slope , str(i+1))
                    else:
                        self.set_right_of_line(self.is_right_negative_zero_slope , str(i+1))
                        
                self.pipeline_function = self.pipeline_with_tracking
                self.set_counting_function(self.count_track_two_lines)
                
            
            else:
                raise ValueError("Please keep sure you have chosen one of the three available counting approaches via one of the following strings: detection_only_approach, tracking_without_line or tracking_with_line")
                
                    
        except ValueError as e:
            print("Error:", e)
            print("Please ensure the arguments are correctly specified according to the provided information.")
            sys.exit(1)

    def preprocess_included_box_mask(self , args):
    
        x_ib1 , y_ib1 = int(args.included_box[0] * self.video_attributes["width"]) , int(args.included_box[1] * self.video_attributes["height"])
        x_ib2 , y_ib2 = int(args.included_box[2] * self.video_attributes["width"]) , int(args.included_box[3] * self.video_attributes["height"])
        self.counting_preprocess["included_box"] = [x_ib1 , y_ib1 , x_ib2 , y_ib2]
        
        for i in range(len(self.counting_preprocess["included_box"])):
            if self.counting_preprocess["included_box"][i] < 0:
                if i==0 or i==2:
                    self.counting_preprocess["included_box"][i] = int(self.video_attributes["width"] +  self.counting_preprocess["included_box"][i])
                if i==1 or i==3:
                    self.counting_preprocess["included_box"][i] = int(self.video_attributes["height"] +  self.counting_preprocess["included_box"][i])
        

    def slope_intercept_with_mask(self , line_num):

        # Points coordinates with pourcentege 
        x1, y1 = self.counting_attributes["line_point"+str(line_num)+"1"]
        x2, y2 = self.counting_attributes["line_point"+str(line_num)+"2"]

        # Calculating new points coordinates with repect to pixels and included bow in case of using a mask
        x_ib1 , y_ib1 = self.counting_preprocess["included_box"][0] , self.counting_preprocess["included_box"][1]
        x_ib2 , y_ib2 = self.counting_preprocess["included_box"][2] , self.counting_preprocess["included_box"][3]
        x1 , y1 = int(x_ib1 + x1 * (x_ib2 - x_ib1)) , int(y_ib1 + y1 * (y_ib2 - y_ib1))
        x2 , y2 = int(x_ib1 + x2 * (x_ib2 - x_ib1)) , int(y_ib1 + y2 * (y_ib2 - y_ib1))
        self.counting_attributes["line_point"+str(line_num)+"1"] = (x1 , y1)
        self.counting_attributes["line_point"+str(line_num)+"2"] = (x2 , y2)

        #calculating slope/intercept
        if x1 == x2:   # Vertical line
            self.counting_attributes["slope"+str(line_num)] = "inf"
            self.counting_attributes["intercept_line"+str(line_num)]  = x1
        else:
            self.counting_attributes["slope"+str(line_num)] = (y2 - y1) / (x2 - x1)
            self.counting_attributes["intercept_line"+str(line_num)] = y1 - self.counting_attributes["slope"+str(line_num)] * x1

        # assign the appropriate ditance function whether the user commands a vertical, horizontal or oblic line.
        if self.counting_attributes["slope1"] == "inf":
            self.set_distance_function(self.dist_v_bbox_line)
        elif self.counting_attributes["slope1"] == 0:
            self.set_distance_function(self.dist_h_bbox_line)                  
        elif self.counting_attributes["slope1"]:
            self.set_distance_function(self.dist_s_bbox_line)      
    
    
    def slope_intercept_without_mask(self , line_num):

        # Points coordinates with pourcentege 
        x1, y1 = self.counting_attributes["line_point"+str(line_num)+"1"]
        x2, y2 = self.counting_attributes["line_point"+str(line_num)+"2"]

        # Calculating new points coordinates with repect to pixels
        x1 , y1 = int(x1 * self.video_attributes["width"]) , int(y1 * self.video_attributes["height"])
        x2 , y2 = int(x2 * self.video_attributes["width"]) , int(y2 * self.video_attributes["height"]) 
        self.counting_attributes["line_point"+str(line_num)+"1"] , self.counting_attributes["line_point"+str(line_num)+"2"] = (x1 , y1) , (x2 , y2)

        #calculating slope/intercept
        if x1 == x2:   # Vertical line
            self.counting_attributes["slope"+str(line_num)] = "inf"
            self.counting_attributes["intercept_line"+str(line_num)] = x1
        else:
            self.counting_attributes["slope"+str(line_num)] = (y2 - y1) / (x2 - x1)
            self.counting_attributes["intercept_line"+str(line_num)] = y1 - self.counting_attributes["slope"+str(line_num)] * x1

        # assign the appropriate ditance function whether thye user commands a vertical, horizontal or oblic line.
        if self.counting_attributes["slope1"] == "inf":
            self.set_distance_function(self.dist_v_bbox_line)
        elif self.counting_attributes["slope1"] == 0:
            self.set_distance_function(self.dist_h_bbox_line)                  
        elif self.counting_attributes["slope1"]:
            self.set_distance_function(self.dist_s_bbox_line)      
    

    
    def get_video_info(self, source):
        
        video_path = os.path.join(os.getcwd(), source)
        video_capture = cv2.VideoCapture(video_path)
        
        # Get the video properties
        self.video_attributes["width"] = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_attributes["height"] = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_attributes["frame_rate"] = int(video_capture.get(cv2.CAP_PROP_FPS))
        self.video_attributes["total_frames"] = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
        video_capture.release()
        
    def preprocess_for_counting(self , im0s):

        def mask_list_image(im0s, included_box):
            """
            Mask images in the list im0s based on the included_box.
            Parameters:
            - im0s (list of numpy arrays): List of images.
            - included_box (tuple): box to keep after masking the input images. It is represented by coordinates (x1, y1, x2, y2).
            Returns:
            - List of masked images.
            """
            masked_images = []
            for img in im0s:
                x1, y1, x2, y2 = self.counting_preprocess["included_box"]
                # Create a binary mask
                mask = np.ones_like(img)
                # Update the mask to set pixels outside the included_box to 0
                mask[:y1, : , :] = 0  # Set pixels above the box to 0
                mask[y2: , : , :] = 0  # Set pixels below the box to 0
                mask[:, :x1, :] = 0  # Set pixels to the left of the box to 0
                mask[:, x2:, :] = 0  # Set pixels to the right of the box to 0
                # Apply the mask to cover pixels outside the included_box with black
                masked_img = img * mask
                masked_images.append(masked_img)
            return masked_images

        if self.counting_preprocess["use_mask"]:
              if self.counting_preprocess["visualize_masked_frames"]:
                  im0s = mask_list_image(im0s , self.counting_preprocess["included_box"] )
                  im = self.predictor.preprocess(im0s)
              else:
                  im0s_ = mask_list_image(im0s , self.counting_preprocess["included_box"] )
                  im = self.predictor.preprocess(im0s_)
        else:
              im = self.predictor.preprocess(im0s)

        return im , im0s


    def postprocess_for_counting(self , preds , im , im0s , path):

            if isinstance(self.predictor.model, AutoBackend):
                self.predictor.results = self.predictor.postprocess(preds, im, im0s)
            else:
                self.predictor.results = self.predictor.model.postprocess(path, preds, im, im0s)



  

    def dist_h_bbox_line(self , bbox, intercept , slope = 0):
        """
        bbox: Tensor of shape (m,2): m consists of the number of bounding boxes in a single frame.
        """
        y = bbox[:,1]
        return abs(y-intercept)

    def dist_v_bbox_line(self , bbox, intercept , slope = None):
        """
        bbox: Tensor of shape (m,2): m consists of the number of bounding boxes in a single frame.
        """
        x = bbox[:,0]
        return abs(x-intercept)

    def dist_s_bbox_line(self , bbox, intercept , slope):
        """
        bbox: Tensor of shape (m,2): m consists of the number of bounding boxes in a single frame.
        """
        x, y = bbox[:,0] , bbox[:,1]
        A = slope
        B = -1
        C = intercept        
        numerator = abs(A * x + B * y + C)
        denominator = np.sqrt(A**2 + B**2)
        return numerator / denominator


    def set_distance_function(self, distance_function):
        self.distance_function = distance_function


    def is_right_vertical(self , bbox , intercept , slope = "inf"):
        return bbox[:, 0] > intercept
    
    def is_right_positive_slope(self , bbox , intercept , slope):
        return bbox[:, 1] < slope * bbox[:, 0] + intercept
    
    def is_right_negative_zero_slope(self , bbox , intercept , slope):
        return bbox[:, 1] > slope * bbox[:, 0] + intercept

    def set_right_of_line(self, right_of_line , line_num):
        self.right_of_line["line"+str(line_num)] = right_of_line

    
    def count_detect_line(self , boxes , intercept1 , slope1 , line_vicinity , intercept2 = None , slope2 = None):
        
        bboxs = boxes.xywh[: , :2]
        
        if bboxs.numel() > 0:
            
            dist = self.distance_function(bbox = bboxs, intercept = intercept1, slope = slope1) < line_vicinity * torch.max(boxes.xywh[:, 2], boxes.xywh[:, 3])
            
            cls = boxes.cls.cpu().int()
            cls = torch.eye(self.max_cls_index)[cls]
            
            mask = ~dist[:, None].expand(-1, self.max_cls_index)
            masked_cls = torch.masked_fill(cls, mask, 0) 
            self.count_per_class += torch.sum(masked_cls, dim=0)
            self.counter += torch.sum(masked_cls)  


    def count_track_noline(self , boxes , intercept1 = None , slope1 = None , line_vicinity = None , intercept2 = None , slope2 = None):
        
        bboxs = boxes.xywh[: , :2]
        ids = boxes.id
        
        if bboxs.numel() > 0 and ids is not None:
            ids_ = torch.tensor([id_ not in self.ids_set for id_ in ids.numpy()], dtype=torch.bool)
            
            indices = torch.nonzero(ids_)
            ids_to_keep_track = ids[indices]
            
            for id in ids_to_keep_track.view(-1):
                self.ids_set.add(int(id))

            cls = boxes.cls.cpu().int()
            cls = torch.eye(self.max_cls_index)[cls]


            
            mask = ~ids_[:, None].expand(-1, self.max_cls_index )
            masked_ids = torch.masked_fill(cls, mask, 0)
            
            self.count_per_class += torch.sum(masked_ids, dim=0)
            self.counter += torch.sum(masked_ids)  

    
    def count_track_line(self , boxes , intercept1 , slope1 , line_vicinity , intercept2 = None , slope2 = None):
        
        bboxs = boxes.xywh[: , :2]
        ids = boxes.id
        
        if bboxs.numel() > 0 and ids is not None:

            ids_ = torch.tensor([id_ not in self.ids_set for id_ in ids.numpy()], dtype=torch.bool)
            dist = self.distance_function(bbox = bboxs, intercept = intercept1, slope = slope1) < line_vicinity * torch.max(boxes.xywh[:, 2], boxes.xywh[:, 3])

            indices = torch.nonzero(ids_ & dist)
            ids_to_keep_track = ids[indices]
            for id in ids_to_keep_track.view(-1):
                self.ids_set.add(int(id))
                
            cls = boxes.cls.cpu().int()
            cls = torch.eye(self.max_cls_index )[cls]
            
            mask = ~ids_[:, None].expand(-1, self.max_cls_index )
            masked_ids = torch.masked_fill(cls, mask, 0)
            mask = ~dist[:, None].expand(-1, self.max_cls_index )
            masked_dist = torch.masked_fill(cls, mask, 0)
            masked_count = torch.logical_and(masked_dist,masked_ids)
            
            self.count_per_class += torch.sum(masked_count, dim=0)
            self.counter += torch.sum(masked_count)  


    def count_track_two_lines(self , boxes , intercept1 , slope1 , line_vicinity , intercept2 , slope2):
        
        line_vicinity = None
        bboxs = boxes.xywh[: , :2]
        ids = boxes.id
        
        if bboxs.numel() > 0 and ids is not None:
    
            ids_ = torch.tensor([id_ not in self.ids_set for id_ in ids.numpy()], dtype=torch.bool)
            
            right_of_line1_vehicles= self.right_of_line["line1"](bboxs , intercept1 , slope1)
            left_of_line2_vehicles = ~self.right_of_line["line2"](bboxs , intercept2 , slope2)
            vehicules_between_two_lines = right_of_line1_vehicles & left_of_line2_vehicles
    
            indices = torch.nonzero(ids_ & vehicules_between_two_lines)
            ids_to_keep_track = ids[indices]
            for id in ids_to_keep_track.view(-1):
                self.ids_set.add(int(id))
    
            cls = boxes.cls.cpu().int()
            cls = torch.eye(self.max_cls_index)[cls]
    
            mask = ~ids_[:, None].expand(-1, self.max_cls_index )
            masked_ids = torch.masked_fill(cls, mask, 0)
    
            mask = ~vehicules_between_two_lines[:, None].expand(-1, self.max_cls_index )
            masked_between_two_lines = torch.masked_fill(cls, mask, 0)
            
            masked_count = torch.logical_and(masked_between_two_lines,masked_ids)
            
            self.count_per_class += torch.sum(masked_count, dim=0)
            self.counter += torch.sum(masked_count)  

            # Speed tracking
            ids_filtered = ids[vehicules_between_two_lines]
            ids_frame = torch.stack((ids_filtered, torch.full_like(ids_filtered, self.frame_number)), dim=1)
            self.ids_frames = torch.cat((self.ids_frames, ids_frame)).int()
            self.id_to_first_last = self.id_to_first_last_frame(self.ids_frames)

    
    def id_to_first_last_frame(self , ids_frames):
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
        self.counting_function = counting_function

    
    
    def run_counting(self , i=0):
        """
        Count the number of vehicles in a video scene (or in a batch of video scenes).

        Args:
            predictor (object): *The predictor object that holds information about the detected and tracked objects in each frame.
                                * predictor.results: holds the results of detection and tracking of the current frame (bounding boxes, ID of each tracked object)
                                *predictor.counting: The counting function (Many approaches and algorithms can be customized)
                                *This object also contains information about the counting method to be used and some parameters of counting.
            i: The index of the video scene in the batch to consider. If you are only using one scene video, please assign 0 to "i."

        """

        line_vicinity = self.counting_attributes["line_vicinity"] 
        intercept1 , slope1 = self.counting_attributes["intercept_line1"] , self.counting_attributes["slope1"]
        intercept2 , slope2 = self.counting_attributes["intercept_line2"] , self.counting_attributes["slope2"]
        boxes = self.predictor.results[i].boxes
        self.counting_function(boxes, intercept1 , slope1 , line_vicinity , intercept2 , slope2)




    def pipeline_with_tracking(self , im0s , path , profilers):

        # Preprocessing
        with profilers[0]:
            im , im0s = self.preprocess_for_counting(im0s)
        # Inference (Detection)
        with profilers[1]:
            preds = self.predictor.inference(im)
        # Postprocessing
        with profilers[2]:
            self.postprocess_for_counting(preds , im , im0s , path)
        # Tracking
        with profilers[3]:
            self.predictor.run_callbacks('on_predict_postprocess_end')

        return im0s , im , profilers

    def pipeline_without_tracking(self , im0s , path , profilers):

        # Preprocessing
        with profilers[0]:
            im , im0s = self.preprocess_for_counting(im0s)
        # Inference (Detection)
        with profilers[1]:
            preds = self.predictor.inference(im)
        # Postprocessing
        with profilers[2]:
            self.postprocess_for_counting(preds , im , im0s , path)
        with profilers[3]:
            pass

        return im0s , im , profilers

    def set_pipeline_function(self, pipeline_function):
        self.pipeline_function = pipeline_function

    def run_pipeline(self , im0s , path , profilers):
        return self.pipeline_function(im0s , path , profilers)





    
    # result = yolo.predictor.results[i]
    # A method of "result" class not yolo.predictor
    def plot(self,idx,counter,count_per_class,line_point11,line_point12,line_point21=None,line_point22=None,
            conf=True,line_width=None,font_size=None,font='Arial.ttf',
            pil=False,img=None,im_gpu=None,kpt_radius=5,kpt_line=True,labels=True,boxes=True,masks=True,probs=True,
            **kwargs  # deprecated args TODO: remove support in 8.2
    ):

        if img is None and isinstance(self.predictor.results[idx].orig_img, torch.Tensor):
            img = np.ascontiguousarray(self.predictor.results[idx].orig_img[0].permute(1, 2, 0).cpu().detach().numpy()) * 255

        # Deprecation warn TODO: remove in 8.2
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


        # Plot Detect results
        if pred_boxes and show_boxes:
            for d in reversed(pred_boxes):
                c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
                name = ('' if id is None else f'id:{id} ') + names[c]
                label = (f'{name} {conf:.2f}' if conf else name) if labels else None
                annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))


        # Plot the Counter
        annotator.text([10,400] , f"counter : {counter}" , txt_color=(255, 255, 255))
        annotator.text([10,200] , f"count per class : {count_per_class}" , txt_color=(255, 255, 255))

        # Plot Classify results
        if pred_probs is not None and show_probs:
            text = ',\n'.join(f'{names[j] if names else j} {pred_probs.data[j]:.2f}' for j in pred_probs.top5)
            x = round(self.predictor.results[idx].orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=(255, 255, 255))  # TODO: allow setting colors

        annotator.draw_line_with_two_points(line_point11, line_point12)
        if self.counting_attributes["counting_approach"] == "tracking_with_two_lines":
            annotator.draw_line_with_two_points(line_point21, line_point22)
        
        return annotator.result()





    # This is a method of yolo.predictor objects
    def write_results(self, idx, results, batch):
        """Write inference results to a file or directory."""
        p, im, _ = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.predictor.source_type.webcam or self.predictor.source_type.from_img or self.predictor.source_type.tensor:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.predictor.dataset.count
        else:
            frame = getattr(self.predictor.dataset, 'frame', 0)
        self.predictor.data_path = p
        self.predictor.txt_path = str(self.predictor.save_dir / 'labels' / p.stem) + ('' if self.predictor.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        result = results[idx]

        log_string += result.verbose()

        if self.predictor.args.save or self.predictor.args.show:  # Add bbox to image

            plot_args = {
                'idx' : idx,
                'counter' : self.counter,
                'count_per_class' : self.count_per_class,
                'line_point11' : self.counting_attributes["line_point11"],
                'line_point12' : self.counting_attributes["line_point12"],
                'line_point21' : self.counting_attributes["line_point21"],
                'line_point22' : self.counting_attributes["line_point22"],
                'line_width': self.predictor.args.line_width,
                'boxes': self.predictor.args.show_boxes,
                'conf': self.predictor.args.show_conf,
                'labels': self.predictor.args.show_labels}

            if not self.predictor.args.retina_masks:
                plot_args['im_gpu'] = im[idx]

            self.predictor.plotted_img = self.plot(**plot_args)

        # Write
        if self.predictor.args.save_txt:
            result.save_txt(f'{self.predictor.txt_path}.txt', save_conf=self.predictor.args.save_conf)
        if self.predictor.args.save_crop:
            result.save_crop(save_dir=self.predictor.save_dir / 'crops',
                              file_name=self.predictor.data_path.stem + ('' if self.predictor.dataset.mode == 'image' else f'_{frame}'))



        return log_string
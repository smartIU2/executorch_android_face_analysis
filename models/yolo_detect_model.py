from copy import deepcopy
from pathlib import Path
import torch
import types
from ultralytics import YOLO
from ultralytics.nn.modules.head import Pose, Pose26
from urllib.request import urlretrieve


YOLO_FACE_URL = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/"


def YOLOModel(yolo_model):
    '''
        Creates a YOLO face detection model with modified output to facilitate analyzing and passing it to the OpenCV NMS function
        from Tensor(1,5,x) with rectangular bounding boxes (x_center, y_center, width, height) and concatenated confidences
         to (Tensor(1,3,x), Tensor(1,1,x)) with square bounding boxes (left, top, inradius) and separate confidences.
             
        When choosing YOLO8 or YOLO26, inference speed is increased by removing keypoint detections from Pose modules.
    '''

    # get the pre-trained YOLO face detection model
    if not Path(yolo_model).is_file():
        print("Downloading YOLO model")
        urlretrieve(YOLO_FACE_URL + yolo_model, yolo_model)
    
    model = YOLO(yolo_model)

    #make sure tensors are fixed
    model = deepcopy(model).to("cpu")

    detectModel = model.model.model[-1]

    #modify output
    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """Decode bounding boxes from predictions."""
        dim = 1
        
        lt, rb = bboxes.chunk(2, dim)
        x1y1 = anchors - lt
        x2y2 = anchors + rb
        c_xy = (x1y1 + x2y2) / 2
        
        # convert all bounding boxes to squares to facilitate further analysis
        wh = x2y2 - x1y1
        wh = torch.max(wh, 1, keepdim=True)[0]
        xy = c_xy - (wh / 2)
        return torch.cat([xy, wh], dim)  # x,y,inradius

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        #no keypoints
        
        dbox = self._get_decode_boxes(x)
        return (dbox, x["scores"].sigmoid()) #no concat
    
        
    detectModel.decode_bboxes = types.MethodType(decode_bboxes, detectModel)
    detectModel._inference = types.MethodType(_inference, detectModel)
    
    
    if isinstance(detectModel, Pose26):
        
        #disable YOLO26 end2end mode (not supported by delegates)
        model.end2end = False
        
        #remove unnecessary layers from Pose26 module
        del detectModel.kpt_shape
        del detectModel.nk
        del detectModel.cv4
        del detectModel.one2one_cv2
        del detectModel.one2one_cv3
        del detectModel.one2one_cv4
        del detectModel.flow_model
        del detectModel.cv4_kpts
        del detectModel.cv4_sigma
        del detectModel.one2one_cv4_kpts
        del detectModel.one2one_cv4_sigma
            
        #revert inference from Pose to Detect model
        def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
            preds = self.forward_head(x, self.cv2, self.cv3, None, None, None)
            return self._inference(preds)
            
        detectModel.forward = types.MethodType(forward, detectModel)
        
    elif isinstance(detectModel, Pose): #Yolo8

        #remove unnecessary layers from Pose module
        del detectModel.kpt_shape
        del detectModel.nk
        del detectModel.cv4
    
        #revert inference from Pose to Detect model
        def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
            preds = self.forward_head(x, self.cv2, self.cv3, None)
            return self._inference(preds)
            
        detectModel.forward = types.MethodType(forward, detectModel)
        
     
    #speed up inference
    model.fuse()

    return model.model
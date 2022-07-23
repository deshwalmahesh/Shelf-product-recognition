from utils.datasets import *
from utils.utils import *


class YOLODet:
    def __init__(self):
        '''
        Class to build a YOLOv5 detector
        '''
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.half = self.device != 'cpu'  # half precision only supported on CUDA


    def build_model(self,weights:str, imgsz:int = 640):
        '''
        Build model from pre trained weights
        args:
            weights: Path to the pre trained weights
            imgsz: YOLOv5 version to load based on image size
        '''
        # google_utils.attempt_download(weights)
        self.model = torch.load(weights, map_location=self.device)['model'].float()  # load to FP32
        self.model.to(self.device).eval()
        
        self.imgsz = check_img_size(imgsz, s=self.model.model[-1].stride.max())  # check img_size
        if self.half: self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

        
        img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device != 'cpu' else None  # run once


    def detect(self, source, conf_thres:float = 0.4, iou_thres:bool = 0.5, agnostic_nms:bool = False, save_txt:str = False, draw_bb:bool = False):
        '''
        Detect Bounding Boxes
        args:
            source: image path or numpy array of shape [Width, height, 3] specificlly in BGR format
            conf_threshold: Min threshold to consider for classification consideration
            iou_thres: Threshold  for NMS
            agnostic_nms: Whether to apply agnostic NMS
            save_txt: whether to save the results in a txt file
            draw_bb: Whether to draw a bounding box for detections on the original image
        '''
        # load image 
        img, im0 = self.load_image(source)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3: img = img.unsqueeze(0) # add a batch dimension

        # Inference
        with torch.no_grad():
            pred = self.model(img, augment=False)[0] # get results
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic = agnostic_nms) # Apply NMS

            det = pred[0]  # detections per img
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                if not (draw_bb or save_txt): return det.detach().cpu().numpy()

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_txt + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if draw_bb:  # Add bbox to image
                        plot_one_box(xyxy, im0, label="", color=self.colors[int(cls)], line_thickness=2)
            
                return im0 if draw_bb else det.detach().cpu().numpy()
            
            return None

    
    def load_image(self, img0):
        '''
        Load image for the model. Either path to string or the BGR numpy array
        '''
        if isinstance(img0, str):
            img0 = cv2.imread(img0)  # BGR
        
        assert img0 is not None, 'Image Not Found '

        img = letterbox(img0, new_shape=self.imgsz)[0] # Padded resize

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img, img0


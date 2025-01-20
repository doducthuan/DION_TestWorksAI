import hydra
import torch
import cv2
import os
import numpy as np
from pathlib import Path
from random import randint
from sort import *
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from datetime import datetime
import pytz

tracker = None

def init_tracker():
    global tracker
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    tracker = Sort(max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=sort_iou_thresh)

rand_color_list = []
    
def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        id = int(identities[i]) if identities is not None else 0
        box_center = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = str(id)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 253), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
    return img

def random_color_list():
    global rand_color_list
    rand_color_list = []
    for i in range(0,5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)

class DetectionPredictor(BasePredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.xml_data = []
        self.frame_count = 0
        self.video_done = False
        self.frequency = 3
        self.size = 0
        self.fps = 0
        self.frame_interval = 0
        self.min_object_size = 10
        self.video_name = Path(str(cfg.source)).stem
        self.save_frames_dir = Path('runs/frames') / self.video_name
        self.save_frames_dir.mkdir(parents=True, exist_ok=True)       

    
        
        
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                      self.args.conf,
                                      self.args.iou,
                                      agnostic=self.args.agnostic_nms,
                                      max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        self.fps = int(self.dataset.cap.get(cv2.CAP_PROP_FPS))
        self.frame_interval = self.fps / self.frequency 
        
        if im0 is None or im is None:
            self.video_done = True
            return ""
        if self.frame_count % self.frame_interval != 0:           
            self.frame_count += 1
            return ""
        else: 
            self.size += 1              
            log_string = ""
            if len(im.shape) == 3:
                im = im[None]
            self.seen += 1
            im0 = im0.copy()
            if self.webcam:
                log_string += f'{idx}: '
                frame = self.dataset.count
            else:
                frame = getattr(self.dataset, 'frame', 0)
    
            self.data_path = p
            save_path = str(self.save_dir / p.name)
            self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
            log_string += '%gx%g ' % im.shape[2:]
            self.annotator = self.get_annotator(im0)
            
            det = preds[idx]
            self.all_outputs.append(det)
            if len(det) == 0:
                return log_string
                
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()
                log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
            
            filtered_dets = []
            for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                width = x2 - x1
                height = y2 - y1
                if width >= self.min_object_size and height >= self.min_object_size:
                    filtered_dets.append([x1, y1, x2, y2, conf, detclass])
            
            dets_to_sort = np.array(filtered_dets) if filtered_dets else np.empty((0,6))
            
            tracked_dets = tracker.update(dets_to_sort)
            tracks = tracker.getTrackers()
            
            # for track in tracks:
            #     [cv2.line(im0, (int(track.centroidarr[i][0]),
            #                    int(track.centroidarr[i][1])), 
            #                   (int(track.centroidarr[i+1][0]),
            #                    int(track.centroidarr[i+1][1])),
            #                    rand_color_list[track.id], thickness=3) 
            #      for i,_ in enumerate(track.centroidarr) 
            #      if i < len(track.centroidarr)-1]
            
            if len(tracked_dets)>0:
                bbox_xyxy = tracked_dets[:,:4]
                identities = tracked_dets[:, 8]
                categories = tracked_dets[:, 4]
                
                #draw_boxes(im0, bbox_xyxy, identities, categories, self.model.names)
                
                frame_filename = f'frame_{self.frame_count:04d}.jpg'
                frame_path = self.save_frames_dir / frame_filename
                cv2.imwrite(str(frame_path), im0)
                
                frame_data = {
                    'id': self.frame_count,
                    'name': frame_filename,
                    'width': im0.shape[1],
                    'height': im0.shape[0],
                    'boxes': []
                }
                
                for box_id, (box, identity) in enumerate(zip(bbox_xyxy, identities)):
                    x1, y1, x2, y2 = box
                    box_data = {
                        'label': 'Vehicle',
                        'xtl': f"{x1:.2f}",
                        'ytl': f"{y1:.2f}",
                        'xbr': f"{x2:.2f}",
                        'ybr': f"{y2:.2f}",
                        'track_id': str(int(identity))
                    }
                    frame_data['boxes'].append(box_data)
                
                self.xml_data.append(frame_data)
                
            self.frame_count += 1
            return log_string

    def export_to_xml(self, output_path):
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        kr_tz = pytz.timezone('Asia/Seoul')
        current_time = datetime.now(kr_tz).strftime('%Y-%m-%d %H:%M:%S.%f%z')
        
        print(f"Saving XML to: {output_path}")
        print(f"Number of frames processed: {len(self.xml_data)}")
        
        xml_content = '<?xml version="1.0" encoding="utf-8"?>\n<annotations>\n'
        xml_content += f'''  <version>1.1</version>
            <meta>
              <task>
                <id>331849</id>
                <job_id>330713</job_id>
                <project>현대자동차_도로이상상황_Tracking</project>
                <name>{self.video_name}</name>
                <size>{self.size}</size>
                <mode>annotation</mode>
                <overlap>0</overlap>
                <bugtracker></bugtracker>
                <created>{current_time}</created>
                <updated>{current_time}</updated>
                <subset>default</subset>
                <start_frame>0</start_frame>
                <stop_frame>{self.size - 1}</stop_frame>
                <frame_filter></frame_filter>
                <segments>
                  <segment>
                    <id>330713</id>
                    <start>0</start>
                    <stop>{self.size - 1}</stop>
                    <url>http://bo-worker.testworks.ai/?id=330713</url>
                  </segment>
                </segments>
                <owner>
                  <username>aw0076</username>
                  <email>ymchae@testworks.co.kr</email>
                </owner>
                <assignee></assignee>
                <labels>
                  <label>
                    <name>Vehicle</name>
                    <color>#d0021b</color>
                    <attributes>
                      <attribute>
                        <name>TRACK_ID</name>
                        <mutable>False</mutable>
                        <input_type>increment</input_type>
                        <default_value>0</default_value>
                        <values>0</values>
                      </attribute>
                    </attributes>
                  </label>
                </labels>
              </task>
              <dumped>{current_time}</dumped>
            </meta>\n'''

        for frame in self.xml_data:
            xml_content += f'''  <image id="{frame['id']}" name="{frame['name']}" width="{frame['width']}" height="{frame['height']}">\n'''
            for box in frame['boxes']:
                xml_content += f'''    <box label="{box['label']}" occluded="0" source="manual" xtl="{box['xtl']}" ytl="{box['ytl']}" xbr="{box['xbr']}" ybr="{box['ybr']}" z_order="0">
                    <attribute name="TRACK_ID">{box['track_id']}</attribute>
                </box>\n'''
            xml_content += '  </image>\n'
        
        xml_content += '</annotations>'
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)
            print(f"Successfully saved XML file with {len(self.xml_data)} frames")
        except Exception as e:
            print(f"Error saving XML file: {str(e)}")

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    init_tracker()
    random_color_list()
    
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    #add
    source_path = Path(cfg.source) if cfg.source is not None else ROOT / "assets"
    video_files = []

    if source_path.is_dir():
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        for ext in video_extensions:
            video_files.extend(source_path.glob(f'*{ext}'))
    else:
        video_files = [source_path]
    # old
    
    for video_file in video_files:
        print(f"\nProcessing video: {video_file.name}")
        
        # Tạo predictor mới cho mỗi video
        cfg.source = str(video_file)
        predictor = DetectionPredictor(cfg)
        
        try:
            predictor()
        except StopIteration:
            print(f"Completed processing {video_file.name}")
        except Exception as e:
            print(f"Error processing {video_file.name}: {str(e)}")
        finally:
            output_dir = Path('runs/result_xml')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\nExporting results for {video_file.name}...")
            print(f"Output directory: {output_dir.absolute()}")
            print(f"Frames saved to: {predictor.save_frames_dir / video_file.stem}")
            print(f"Total frames saved: {len(list(predictor.save_frames_dir.glob('*.jpg')))}")
            
            output_path = output_dir / f'{video_file.stem}.xml'
            predictor.export_to_xml(str(output_path))
            
            if hasattr(predictor.dataset, 'cap'):
                predictor.dataset.cap.release()
            cv2.destroyAllWindows()
            
    print("\nAll videos processed successfully!")

if __name__ == "__main__":
    predict()
### Steps to run Code
- Note : use python 3.11.1
- Clone the repository
```
https://github.com/doducthuan/DION_TestWorksAI.git
```

- Goto cloned folder
```
cd DION_TestWorksAI
```

- Install the ultralytics package
```
pip install ultralytics==8.0.0
```

- Do Tracking with mentioned command below
```
#add data
Note : Delete file please_delete_this_file_before_adding_the_video.txt ( Require )
B1 : Add all the videos you want to train into folder input_videos

#run
B2: python yolo\v8\detect\detect_and_trk.py model=yolov8s.pt source="path to folder input_videos" show=False
```

- References 
```
https://github.com/RizwanMunawar/yolov8-object-tracking
```

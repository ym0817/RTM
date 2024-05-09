python tools/deploy.py \
    configs/mmdet/detection/detection_onnxruntime_static.py \
    ../mmpose/projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \
    demo/resources/human-pose.jpg \
    --work-dir rtmpose-ort/rtmdet-nano \
    --device cpu \
    --show \
    --dump-info  # dump sdk info



    #https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \
    #https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
    #--input webcam
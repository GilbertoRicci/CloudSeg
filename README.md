# CloudSeg
Clouds semantic segmentation in horizontal sky images

# Requirements
python                    3.10.11
paddleseg                 2.8.0
ffmpeg                    4.3.1
numpy                     1.24.3
opencv-python             4.5.5.64

# Usage
cloudseg.py [-h] -d DATASET_DIR -i INF_OUT_DIR [-s] -p PS_ROOT_DIR -m MODEL_YAML -v VIDEO_FILE_PATH
                   [-f VIDEO_FRAME_DUR] [-o] [-a ANNOTATION_ALPHA] [-c FOURCC_STRING]
                   
# Options
  -h, --help            show this help message and exit
  -d DATASET_DIR, --dataset_dir DATASET_DIR
                        Dataset directory
  -i INF_OUT_DIR, --inf_out_dir INF_OUT_DIR
                        Inference output directory
  -s, --run_segmentation
                        Run segmentation or just read inference folder annotations
  -p PS_ROOT_DIR, --ps_root_dir PS_ROOT_DIR
                        PaddleSeg root path
  -m MODEL_YAML, --model_yaml MODEL_YAML
                        PPLiteSeg model YAML path
  -v VIDEO_FILE_PATH, --video_file_path VIDEO_FILE_PATH
                        Video output file path
  -f VIDEO_FRAME_DUR, --video_frame_dur VIDEO_FRAME_DUR
                        Video frame duration (in seconds)
  -o, --only_annotation
                        Video w/ annotation only or annotated original image
  -a ANNOTATION_ALPHA, --annotation_alpha ANNOTATION_ALPHA
                        Annotation alpha: value from 0.0 (opaque) to 1.0 (full transparend) (for annotated original image only)
  -c FOURCC_STRING, --fourcc_string FOURCC_STRING
                        FourCC video codec code 

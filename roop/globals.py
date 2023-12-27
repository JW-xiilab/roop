<<<<<<< HEAD
from typing import List, Optional

source_path: Optional[str] = None
target_path: Optional[str] = None
output_path: Optional[str] = None
headless: Optional[bool] = None
frame_processors: List[str] = []
keep_fps: Optional[bool] = None
keep_frames: Optional[bool] = None
skip_audio: Optional[bool] = None
many_faces: Optional[bool] = None
reference_face_position: Optional[int] = None
reference_frame_number: Optional[int] = None
similar_face_distance: Optional[float] = None
temp_frame_format: Optional[str] = None
temp_frame_quality: Optional[int] = None
output_video_encoder: Optional[str] = None
output_video_quality: Optional[int] = None
max_memory: Optional[int] = None
execution_providers: List[str] = []
execution_threads: Optional[int] = None
log_level: str = 'error'
=======
from settings import Settings
from typing import List

source_path = None
target_path = None
output_path = None
target_folder_path = None

frame_processors: List[str] = []
keep_fps = None
keep_frames = None
skip_audio = None
wait_after_extraction = None
many_faces = None
use_batch = None
source_face_index = 0
target_face_index = 0
face_position = None
video_encoder = None
video_quality = None
max_memory = None
execution_providers: List[str] = []
execution_threads = None
headless = None
log_level = 'error'
selected_enhancer = None
face_swap_mode = None
blend_ratio = 0.5
distance_threshold = 0.65
default_det_size = True

no_face_action = 0

processing = False 

FACE_ENHANCER = None

INPUT_FACESETS = []
TARGET_FACES = []

IMAGE_CHAIN_PROCESSOR = None
VIDEO_CHAIN_PROCESSOR = None
BATCH_IMAGE_CHAIN_PROCESSOR = None

CFG: Settings = None


>>>>>>> a55b78d4ec1eb6c48da7d36c1954797959a34f37

from typing import Any, List, Callable
import cv2
import insightface
import threading

import roop.globals
import roop.processors.frame.core
<<<<<<< HEAD
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
=======
from roop.face_util import get_first_face, get_all_faces
>>>>>>> a55b78d4ec1eb6c48da7d36c1954797959a34f37
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video, compute_cosine_distance, get_destfilename_from_path

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'

DIST_THRESHOLD = 0.65


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER

    FACE_SWAPPER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
<<<<<<< HEAD
    conditional_download(download_directory_path, ['https://huggingface.co/CountFloyd/deepfake/resolve/main/inswapper_128.onnx'])
=======
    conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx'])
>>>>>>> a55b78d4ec1eb6c48da7d36c1954797959a34f37
    return True


def pre_start() -> bool:
    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


<<<<<<< HEAD
def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
=======
def process_frame(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    global DIST_THRESHOLD

>>>>>>> a55b78d4ec1eb6c48da7d36c1954797959a34f37
    if roop.globals.many_faces:
        many_faces = get_all_faces(temp_frame)
        if many_faces is not None:
            for target_face in many_faces:
                if target_face['det_score'] > 0.65:
                    temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
<<<<<<< HEAD
        target_face = find_similar_face(temp_frame, reference_face)
=======
>>>>>>> a55b78d4ec1eb6c48da7d36c1954797959a34f37
        if target_face:
            target_embedding = target_face.embedding
            many_faces = get_all_faces(temp_frame)
            target_face = None
            for dest_face in many_faces:
                dest_embedding = dest_face.embedding
                if compute_cosine_distance(target_embedding, dest_embedding) <= DIST_THRESHOLD:
                    target_face = dest_face
                    break
            if target_face:
                temp_frame = swap_face(source_face, target_face, temp_frame)
            return temp_frame
                    
        target_face = get_first_face(temp_frame)
        if target_face is not None:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame


<<<<<<< HEAD
def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    reference_face = None if roop.globals.many_faces else get_face_reference()
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, reference_face, temp_frame)
        cv2.imwrite(temp_frame_path, result)
=======

def process_frames(is_batch: bool, source_face: Face, target_face: Face, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is not None:
            result = process_frame(source_face, target_face, temp_frame)
            if result is not None:
                if is_batch:
                    tf = get_destfilename_from_path(temp_frame_path, roop.globals.output_path, '_fake.png')
                    cv2.imwrite(tf, result)
                else:
                    cv2.imwrite(temp_frame_path, result)
>>>>>>> a55b78d4ec1eb6c48da7d36c1954797959a34f37
        if update:
            update()


def process_image(source_face: Any, target_face: Any, target_path: str, output_path: str) -> None:
    global DIST_THRESHOLD

    target_frame = cv2.imread(target_path)
<<<<<<< HEAD
    reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
    result = process_frame(source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)
=======
    if target_frame is not None:
        result = process_frame(source_face, target_face, target_frame)
        if result is not None:
            cv2.imwrite(output_path, result)


def process_video(source_face: Any, target_face: Any, temp_frame_paths: List[str]) -> None:
    global DIST_THRESHOLD

    roop.processors.frame.core.process_video(source_face, target_face, temp_frame_paths, process_frames)

>>>>>>> a55b78d4ec1eb6c48da7d36c1954797959a34f37

def process_batch_images(source_face: Any, target_face: Any, temp_frame_paths: List[str]) -> None:
    global DIST_THRESHOLD

<<<<<<< HEAD
def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not roop.globals.many_faces and not get_face_reference():
        reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
        reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
        set_face_reference(reference_face)
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
=======
    roop.processors.frame.core.process_batch(source_face, target_face, temp_frame_paths, process_frames)
>>>>>>> a55b78d4ec1eb6c48da7d36c1954797959a34f37

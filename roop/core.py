#!/usr/bin/env python3

import os
import sys
import shutil
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'

import warnings
from typing import List
import platform
import signal
<<<<<<< HEAD
import shutil
import argparse
import onnxruntime
import tensorflow
import roop.globals
import roop.metadata
import roop.ui as ui
from roop.predictor import predict_image, predict_video
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path
=======
import torch
import onnxruntime
import pathlib

from time import time

import roop.globals
import roop.metadata
import roop.utilities as util
import roop.util_ffmpeg as ffmpeg
import ui.main as main
from settings import Settings
from roop.face_util import extract_face_images
from roop.ProcessEntry import ProcessEntry
from roop.ProcessMgr import ProcessMgr
from roop.ProcessOptions import ProcessOptions
from roop.capturer import get_video_frame_total


clip_text = None

call_display_ui = None

process_mgr = None

>>>>>>> a55b78d4ec1eb6c48da7d36c1954797959a34f37

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
<<<<<<< HEAD
    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('-s', '--source', help='select an source image', dest='source_path')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--frame-processor', help='frame processors (choices: face_swapper, face_enhancer, ...)', dest='frame_processor', default=['face_swapper'], nargs='+')
    program.add_argument('--keep-fps', help='keep target fps', dest='keep_fps', action='store_true')
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true')
    program.add_argument('--skip-audio', help='skip target audio', dest='skip_audio', action='store_true')
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true')
    program.add_argument('--reference-face-position', help='position of the reference face', dest='reference_face_position', type=int, default=0)
    program.add_argument('--reference-frame-number', help='number of the reference frame', dest='reference_frame_number', type=int, default=0)
    program.add_argument('--similar-face-distance', help='face distance used for recognition', dest='similar_face_distance', type=float, default=0.85)
    program.add_argument('--temp-frame-format', help='image format used for frame extraction', dest='temp_frame_format', default='png', choices=['jpg', 'png'])
    program.add_argument('--temp-frame-quality', help='image quality used for frame extraction', dest='temp_frame_quality', type=int, default=0, choices=range(101), metavar='[0-100]')
    program.add_argument('--output-video-encoder', help='encoder used for the output video', dest='output_video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc'])
    program.add_argument('--output-video-quality', help='quality used for the output video', dest='output_video_quality', type=int, default=35, choices=range(101), metavar='[0-100]')
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int)
    program.add_argument('--execution-provider', help='available execution provider (choices: cpu, ...)', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version', version=f'{roop.metadata.name} {roop.metadata.version}')

    args = program.parse_args()

    roop.globals.source_path = args.source_path
    roop.globals.target_path = args.target_path
    roop.globals.output_path = normalize_output_path(roop.globals.source_path, roop.globals.target_path, args.output_path)
    roop.globals.headless = roop.globals.source_path is not None and roop.globals.target_path is not None and roop.globals.output_path is not None
    roop.globals.frame_processors = args.frame_processor
    roop.globals.keep_fps = args.keep_fps
    roop.globals.keep_frames = args.keep_frames
    roop.globals.skip_audio = args.skip_audio
    roop.globals.many_faces = args.many_faces
    roop.globals.reference_face_position = args.reference_face_position
    roop.globals.reference_frame_number = args.reference_frame_number
    roop.globals.similar_face_distance = args.similar_face_distance
    roop.globals.temp_frame_format = args.temp_frame_format
    roop.globals.temp_frame_quality = args.temp_frame_quality
    roop.globals.output_video_encoder = args.output_video_encoder
    roop.globals.output_video_quality = args.output_video_quality
    roop.globals.max_memory = args.max_memory
    roop.globals.execution_providers = decode_execution_providers(args.execution_provider)
    roop.globals.execution_threads = args.execution_threads
=======
    roop.globals.headless = False
    # Always enable all processors when using GUI
    if len(sys.argv) > 1:
        print('No CLI args supported - use Settings Tab instead')
    roop.globals.frame_processors = ['face_swapper', 'face_enhancer']
>>>>>>> a55b78d4ec1eb6c48da7d36c1954797959a34f37


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        return 8
    return 1


def limit_resources() -> None:
    # limit memory usage
    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = roop.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


<<<<<<< HEAD
=======

def release_resources() -> None:
    import gc
    global process_mgr

    if process_mgr is not None:
        process_mgr.release_resources()
        process_mgr = None

    gc.collect()
    # if 'CUDAExecutionProvider' in roop.globals.execution_providers and torch.cuda.is_available():
    #     with torch.cuda.device('cuda'):
    #         torch.cuda.empty_cache()
    #         torch.cuda.ipc_collect()


>>>>>>> a55b78d4ec1eb6c48da7d36c1954797959a34f37
def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    
    download_directory_path = util.resolve_relative_path('../models')
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/GFPGANv1.4.onnx'])
    util.conditional_download(download_directory_path, ['https://github.com/csxmli2016/DMDNet/releases/download/v1/DMDNet.pth'])
    util.conditional_download(download_directory_path, ['https://github.com/facefusion/facefusion-assets/releases/download/models/GPEN-BFR-512.onnx'])

    download_directory_path = util.resolve_relative_path('../models/CLIP')
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/rd64-uni-refined.pth'])
    download_directory_path = util.resolve_relative_path('../models/CodeFormer')
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/CodeFormerv0.1.onnx'])

    if not shutil.which('ffmpeg'):
       update_status('ffmpeg is not installed.')
    return True

def set_display_ui(function):
    global call_display_ui

    call_display_ui = function


def update_status(message: str) -> None:
    global call_display_ui

    print(message)
    if call_display_ui is not None:
        call_display_ui(message)




def start() -> None:
<<<<<<< HEAD
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_start():
            return
    # process image to image
    if has_image_extension(roop.globals.target_path):
        if predict_image(roop.globals.target_path):
            destroy()
        shutil.copy2(roop.globals.target_path, roop.globals.output_path)
        # process frame
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_image(roop.globals.source_path, roop.globals.output_path, roop.globals.output_path)
            frame_processor.post_process()
        # validate image
        if is_image(roop.globals.target_path):
            update_status('Processing to image succeed!')
        else:
            update_status('Processing to image failed!')
        return
    # process image to videos
    if predict_video(roop.globals.target_path):
        destroy()
    update_status('Creating temporary resources...')
    create_temp(roop.globals.target_path)
    # extract frames
    if roop.globals.keep_fps:
        fps = detect_fps(roop.globals.target_path)
        update_status(f'Extracting frames with {fps} FPS...')
        extract_frames(roop.globals.target_path, fps)
    else:
        update_status('Extracting frames with 30 FPS...')
        extract_frames(roop.globals.target_path)
    # process frame
    temp_frame_paths = get_temp_frame_paths(roop.globals.target_path)
    if temp_frame_paths:
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_video(roop.globals.source_path, temp_frame_paths)
            frame_processor.post_process()
    else:
        update_status('Frames not found...')
        return
    # create video
    if roop.globals.keep_fps:
        fps = detect_fps(roop.globals.target_path)
        update_status(f'Creating video with {fps} FPS...')
        create_video(roop.globals.target_path, fps)
    else:
        update_status('Creating video with 30 FPS...')
        create_video(roop.globals.target_path)
    # handle audio
    if roop.globals.skip_audio:
        move_temp(roop.globals.target_path, roop.globals.output_path)
        update_status('Skipping audio...')
    else:
        if roop.globals.keep_fps:
            update_status('Restoring audio...')
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
        restore_audio(roop.globals.target_path, roop.globals.output_path)
    # clean temp
    update_status('Cleaning temporary resources...')
    clean_temp(roop.globals.target_path)
    # validate video
    if is_video(roop.globals.target_path):
        update_status('Processing to video succeed!')
    else:
        update_status('Processing to video failed!')
=======
    if roop.globals.headless:
        print('Headless mode currently unsupported - starting UI!')
        # faces = extract_face_images(roop.globals.source_path,  (False, 0))
        # roop.globals.INPUT_FACES.append(faces[roop.globals.source_face_index])
        # faces = extract_face_images(roop.globals.target_path,  (False, util.has_image_extension(roop.globals.target_path)))
        # roop.globals.TARGET_FACES.append(faces[roop.globals.target_face_index])
        # if 'face_enhancer' in roop.globals.frame_processors:
        #     roop.globals.selected_enhancer = 'GFPGAN'
       
    batch_process(None, False, None)


def get_processing_plugins(use_clip):
    processors = "faceswap"
    if use_clip:
        processors += ",mask_clip2seg"
    
    if roop.globals.selected_enhancer == 'GFPGAN':
        processors += ",gfpgan"
    elif roop.globals.selected_enhancer == 'Codeformer':
        processors += ",codeformer"
    elif roop.globals.selected_enhancer == 'DMDNet':
        processors += ",dmdnet"
    elif roop.globals.selected_enhancer == 'GPEN':
        processors += ",gpen"
    return processors


def live_swap(frame, swap_mode, use_clip, clip_text, selected_index = 0):
    global process_mgr

    if frame is None:
        return frame

    if process_mgr is None:
        process_mgr = ProcessMgr(None)
    
    options = ProcessOptions(get_processing_plugins(use_clip), roop.globals.distance_threshold, roop.globals.blend_ratio, swap_mode, selected_index, clip_text)
    process_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, options)
    newframe = process_mgr.process_frame(frame)
    if newframe is None:
        return frame
    return newframe


def preview_mask(frame, clip_text):
    import numpy as np
    global process_mgr
    
    maskimage = np.zeros((frame.shape), np.uint8)
    if process_mgr is None:
        process_mgr = ProcessMgr(None)
    options = ProcessOptions("mask_clip2seg", roop.globals.distance_threshold, roop.globals.blend_ratio, "None", 0, clip_text)
    process_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, options)
    maskprocessor = next((x for x in process_mgr.processors if x.processorname == 'clip2seg'), None)
    return process_mgr.process_mask(maskprocessor, frame, maskimage)
    




def batch_process(files:list[ProcessEntry], use_clip, new_clip_text, use_new_method, progress) -> None:
    global clip_text, process_mgr

    roop.globals.processing = True
    release_resources()
    limit_resources()

    # limit threads for some providers
    max_threads = suggest_execution_threads()
    if max_threads == 1:
        roop.globals.execution_threads = 1

    imagefiles:list[ProcessEntry] = []
    videofiles:list[ProcessEntry] = []
           
    update_status('Sorting videos/images')


    for index, f in enumerate(files):
        fullname = f.filename
        if util.has_image_extension(fullname):
            destination = util.get_destfilename_from_path(fullname, roop.globals.output_path, f'.{roop.globals.CFG.output_image_format}')
            destination = util.replace_template(destination, index=index)
            pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)
            f.finalname = destination
            imagefiles.append(f)

        elif util.is_video(fullname) or util.has_extension(fullname, ['gif']):
            destination = util.get_destfilename_from_path(fullname, roop.globals.output_path, f'__temp.{roop.globals.CFG.output_video_format}')
            f.finalname = destination
            videofiles.append(f)


    if process_mgr is None:
        process_mgr = ProcessMgr(progress)
    
    options = ProcessOptions(get_processing_plugins(use_clip), roop.globals.distance_threshold, roop.globals.blend_ratio, roop.globals.face_swap_mode, 0, new_clip_text)
    process_mgr.initialize(roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, options)

    if(len(imagefiles) > 0):
        update_status('Processing image(s)')
        origimages = []
        fakeimages = []
        for f in imagefiles:
            origimages.append(f.filename)
            fakeimages.append(f.finalname)

        process_mgr.run_batch(origimages, fakeimages, roop.globals.execution_threads)
        origimages.clear()
        fakeimages.clear()

    if(len(videofiles) > 0):
        for index,v in enumerate(videofiles):
            if not roop.globals.processing:
                end_processing('Processing stopped!')
                return
            fps = v.fps if v.fps > 0 else util.detect_fps(v.filename)
            if v.endframe == 0:
                v.endframe = get_video_frame_total(v.filename)

            update_status(f'Creating {os.path.basename(v.finalname)} with {fps} FPS...')
            start_processing = time()
            if roop.globals.keep_frames or not use_new_method:
                util.create_temp(v.filename)
                update_status('Extracting frames...')
                ffmpeg.extract_frames(v.filename,v.startframe,v.endframe, fps)
                if not roop.globals.processing:
                    end_processing('Processing stopped!')
                    return

                temp_frame_paths = util.get_temp_frame_paths(v.filename)
                process_mgr.run_batch(temp_frame_paths, temp_frame_paths, roop.globals.execution_threads)
                if not roop.globals.processing:
                    end_processing('Processing stopped!')
                    return
                if roop.globals.wait_after_extraction:
                    extract_path = os.path.dirname(temp_frame_paths[0])
                    util.open_folder(extract_path)
                    input("Press any key to continue...")
                    print("Resorting frames to create video")
                    util.sort_rename_frames(extract_path)                                    
                
                ffmpeg.create_video(v.filename, v.finalname, fps)
                if not roop.globals.keep_frames:
                    util.delete_temp_frames(temp_frame_paths[0])
            else:
                if util.has_extension(v.filename, ['gif']):
                    skip_audio = True
                else:
                    skip_audio = roop.globals.skip_audio
                process_mgr.run_batch_inmem(v.filename, v.finalname, v.startframe, v.endframe, fps,roop.globals.execution_threads, skip_audio)
                
            if not roop.globals.processing:
                end_processing('Processing stopped!')
                return
            
            video_file_name = v.finalname
            if os.path.isfile(video_file_name):
                destination = ''
                if util.has_extension(v.filename, ['gif']):
                    gifname = util.get_destfilename_from_path(v.filename, roop.globals.output_path, '.gif')
                    destination = util.replace_template(gifname, index=index)
                    pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)

                    update_status('Creating final GIF')
                    ffmpeg.create_gif_from_video(video_file_name, destination)
                    if os.path.isfile(destination):
                        os.remove(video_file_name)
                else:
                    skip_audio = roop.globals.skip_audio
                    destination = util.replace_template(video_file_name, index=index)
                    pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)

                    if not skip_audio:
                        ffmpeg.restore_audio(video_file_name, v.filename, v.startframe, v.endframe, destination)
                        if os.path.isfile(destination):
                            os.remove(video_file_name)
                    else:
                        shutil.move(video_file_name, destination)
                update_status(f'\nProcessing {os.path.basename(destination)} took {time() - start_processing} secs')

            else:
                update_status(f'Failed processing {os.path.basename(v.finalname)}!')
    end_processing('Finished')


def end_processing(msg:str):
    update_status(msg)
    roop.globals.target_folder_path = None
    release_resources()
>>>>>>> a55b78d4ec1eb6c48da7d36c1954797959a34f37


def destroy() -> None:
    if roop.globals.target_path:
<<<<<<< HEAD
        clean_temp(roop.globals.target_path)
=======
        util.clean_temp(roop.globals.target_path)
    release_resources()        
>>>>>>> a55b78d4ec1eb6c48da7d36c1954797959a34f37
    sys.exit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    roop.globals.CFG = Settings('config.yaml')
    roop.globals.execution_threads = roop.globals.CFG.max_threads
    roop.globals.video_encoder = roop.globals.CFG.output_video_codec
    roop.globals.video_quality = roop.globals.CFG.video_quality
    roop.globals.max_memory = roop.globals.CFG.memory_limit if roop.globals.CFG.memory_limit > 0 else None
    main.run()

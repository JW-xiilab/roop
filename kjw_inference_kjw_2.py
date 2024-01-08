#!/usr/bin/env python3

import os
import sys
import shutil
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import warnings
from typing import List
from pathlib import Path
import platform
import onnxruntime
import pathlib

from time import time

# import roop.metadata
from roop.kjw_utils import global_vars
from roop.kjw_utils.capturer import get_video_frame_total
from roop.kjw_utils.face_util import extract_face_images
from roop.kjw_utils import FaceSet
from roop.kjw_utils import ffmpeg_util as ffmpeg
from roop.kjw_utils import utils as util
from roop.kjw_utils import exec_util

from settings import Settings

from roop.kjw_proc.DataEntry import DataEntry
from roop.kjw_proc.ProcessMgr import ProcessMgr
# from roop.kjw_proc.ProcessOptions import ProcessOptions


# clip_text = None
# call_display_ui = None
# process_mgr = None


warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def update_global_vars(parser):
    for key, value in global_vars.default_values.items():
        parser.add_argument(f'--{key}', default=value)
    return parser

def setup_args(parser:argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('-i', '--img_path', default='/DATA_17/kjw/01-DeepFake/tmp_dataset/src/park_narae.jpg', type=str, help='source image path of face to swap, ideally in image extension')
    parser.add_argument('-v', '--video_path', default='/DATA_17/kjw/01-DeepFake/tmp_dataset/requests/redcarpet_seoyeji.mp4', type=str, help='destination video path to swap face with')
    # parser.add_argument('-e', '--enhancer', default='', type=str, help='Enhancer model, currently only GFPGAN available')
    parser.add_argument('--face_distance', default=0.65, type=float, help='Max Face similiarity Threshold')
    parser.add_argument('--blend_ratio', default=0.5, type=float, help='Original/Enhanced image blend ratio')
    parser.add_argument('--keep_frames', action='store_true', help='either keep the each frame saved')
    parser.add_argument('--outputs', default='')
    parser.add_argument('--custom_fa', action='store_true', help='either to use custom face analyser or use naive')
    parser.add_argument('--enhancer', action='store_true', help='either to use enhancer or not, currently only GFPGAN available')
    
    parser = update_global_vars(parser)
    parser = parser.parse_args()
    # parser.headless = False
    # parser.frame_processors = ['face_swapper', 'face_enhancer']
    return parser


def limit_resources(max_memory) -> None:
    # limit memory usage
    if max_memory:
        memory = max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))

def pre_check() -> bool:
    if sys.version_info < (3, 9):
        util.update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False

    if not shutil.which('ffmpeg'):
    #    util.update_status('ffmpeg is not installed.')
       util.update_status('ffmpeg is not installed.')
    return True


# def get_processing_plugins(use_enhancer):
#     processors = "faceswap"
#     if use_enhancer:
#         processor += ',gfpagan'
#     # processors += ",gfpgan"
#     # if use_clip:
#     #     processors += ",mask_clip2seg"
    
#     # if roop.globals.selected_enhancer == 'GFPGAN':
#     #     processors += ",gfpgan"
#     # elif roop.globals.selected_enhancer == 'Codeformer':
#     #     processors += ",codeformer"
#     # elif roop.globals.selected_enhancer == 'DMDNet':
#     #     processors += ",dmdnet"
#     # elif roop.globals.selected_enhancer == 'GPEN':
#     #     processors += ",gpen"
#     return processors


def batch_process(args, files:list[DataEntry], use_clip, new_clip_text, use_new_method, progress) -> None:
    # global clip_text, process_mgr

    args.processing = True
    exec_util.release_resources()
    limit_resources(args.max_memory)

    # limit threads for some providers
    max_threads = exec_util.suggest_execution_threads()
    # max_threads = 1
    if max_threads == 1:
        args.execution_threads = 1

    imagefiles:list[DataEntry] = []
    videofiles:list[DataEntry] = []
           
    util.update_status('Sorting videos/images')

    for index, f in enumerate(files):
        fullname = f.filename
        if util.has_image_extension(fullname):
            destination = util.get_destfilename_from_path(fullname, args.output_path, f'.{args.CFG.output_image_format}')
            destination = util.replace_template(args, destination, index=index)
            pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)
            f.finalname = destination
            imagefiles.append(f)

        elif util.is_video(fullname) or util.has_extension(fullname, ['gif']):
            destination = util.get_destfilename_from_path(fullname, args.output_path, f'__temp.{args.CFG.output_video_format}')
            f.finalname = destination
            videofiles.append(f)

    # if process_mgr is None:
    process_mgr = ProcessMgr(args, progress, new_clip_text)
    
    # options = ProcessOptions(util.get_processing_plugins(args.enhancer), args.distance_threshold, args.blend_ratio, args.face_swap_mode, 0, new_clip_text)
    # options = process_mgr.options
    process_mgr.initialize(args.INPUT_FACESETS, args.TARGET_FACES)

    # if(len(imagefiles) > 0):
    #     util.update_status('Processing image(s)')
    #     origimages = []
    #     fakeimages = []
    #     for f in imagefiles:
    #         origimages.append(f.filename)
    #         fakeimages.append(f.finalname)

    #     process_mgr.run_batch(origimages, fakeimages, args.execution_threads)
    #     origimages.clear()
    #     fakeimages.clear()

    if(len(videofiles) > 0):
        for index,v in enumerate(videofiles):
            if not args.processing:
                exec_util.end_processing('Processing stopped!')
                return
            fps = v.fps if v.fps > 0 else util.detect_fps(v.filename)
            if v.endframe == 0:
                v.endframe = get_video_frame_total(v.filename)

            util.update_status(f'Creating {os.path.basename(v.finalname)} with {fps} FPS...')
            start_processing = time()
            if args.keep_frames or not use_new_method:
                util.create_temp(v.filename)
                util.update_status('Extracting frames...')
                # ffmpeg.extract_frames(v.filename,v.startframe,v.endframe, fps, args.CFG.output_image_format)
                if not args.processing:
                    exec_util.end_processing('Processing stopped!', process_mgr)
                    return

                temp_frame_paths = util.get_temp_frame_paths(v.filename, args.CFG.output_image_format)
                process_mgr.run_batch(temp_frame_paths, temp_frame_paths)
                if not args.processing:
                    exec_util.end_processing('Processing stopped!', process_mgr)
                    return
                if args.wait_after_extraction:
                    extract_path = os.path.dirname(temp_frame_paths[0])
                    util.open_folder(extract_path)
                    input("Press any key to continue...")
                    print("Resorting frames to create video")
                    util.sort_rename_frames(extract_path, args.CFG.output_image_format)                            
                
                ffmpeg.create_video(args, v.filename, v.finalname, fps)
                if not args.keep_frames:
                    util.delete_temp_frames(temp_frame_paths[0])
            else:
                if util.has_extension(v.filename, ['gif']):
                    skip_audio = True
                else:
                    skip_audio = args.skip_audio
                process_mgr.run_batch_inmem(v.filename, v.finalname, v.startframe, v.endframe, fps, args.execution_threads, skip_audio)
            
            # TODO: where is "args.processing=False"?
            if not args.processing:
                exec_util.end_processing('Processing stopped!', process_mgr)
                return
            
            video_file_name = v.finalname
            if os.path.isfile(video_file_name):
                destination = ''
                if util.has_extension(v.filename, ['gif']):
                    gifname = util.get_destfilename_from_path(v.filename, args.output_path, '.gif')
                    destination = util.replace_template(args, gifname, index=index)
                    pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)

                    util.update_status('Creating final GIF')
                    ffmpeg.create_gif_from_video(video_file_name, destination)
                    if os.path.isfile(destination):
                        os.remove(video_file_name)
                else:
                    skip_audio = args.skip_audio
                    destination = util.replace_template(args, video_file_name, index=index)
                    pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)

                    if not skip_audio:
                        ffmpeg.restore_audio(video_file_name, v.filename, v.startframe, v.endframe, destination)
                        if os.path.isfile(destination):
                            os.remove(video_file_name)
                    else:
                        shutil.move(video_file_name, destination)
                util.update_status(f'\nProcessing {os.path.basename(destination)} took {time() - start_processing} secs')

            else:
                util.update_status(f'Failed processing {os.path.basename(v.finalname)}!')
    exec_util.end_processing('Finished', process_mgr)


def prepare_video(destfiles, list_files_process:list[DataEntry] = []):
    idx = 0
    # for f in destfiles:
    list_files_process.append(DataEntry(str(destfiles), 0,0, 0))
    filename = list_files_process[idx].filename
    if util.is_video(filename) or filename.lower().endswith('gif'):
        total_frames = get_video_frame_total(filename)
        if list_files_process[idx].endframe == 0:
            list_files_process[idx].endframe = total_frames 
    else:
        total_frames = 0
    return list_files_process

def swap(args, video_files, clip_text=None, use_clip=False):
    if args.CFG.clear_output:
        shutil.rmtree(args.output_path)

    args.selected_enhancer = args.enhancer
    args.target_path = None  # TODO:?
    args.distance_threshold = args.face_distance
    # args.blend_ratio = args.blend_ratio
    # args.keep_frames = args.keep_frames

    # use_clip=False
    # args.execution_threads = args.CFG.max_threads
    batch_process(args, video_files, use_clip, clip_text, True, progress=None)
    args.processing = False


def run() -> None:
    parser = argparse.ArgumentParser(description="FaceSwapper")
    args = setup_args(parser)
    if not pre_check():
        return
    args.frame_processors = ['face_swapper', 'face_enhancer']
    args.CFG = Settings('config.yaml')
    args.CFG.clear_output = False
    args.execution_threads = args.CFG.max_threads
    args.video_encoder = args.CFG.output_video_codec
    args.video_quality = args.CFG.video_quality
    args.max_memory = args.CFG.memory_limit if args.CFG.memory_limit > 0 else None
    args.max_memory = args.CFG.memory_limit if args.CFG.memory_limit > 0 else None
    args.wait_after_extraction = False
    args.skip_audio = False
    args.face_swap_mode = "first"
    args.no_face_action = 0
    args.execution_providers = exec_util.decode_execution_providers([args.CFG.provider])
    # args.custom_fa = args.custom_fa
    # args.keep_frames = True
    print(f'Using provider {args.execution_providers} - Device:{util.get_device(args.execution_providers)}')  
    args.output_path = util.prepare_environment(args, args.outputs)
    assert util.has_image_extension(args.img_path)
    # roop.globals.source_path = args.img_path
    SELECTION_FACES_DATA = extract_face_images(args,  (False, 0))
    args.INPUT_FACESETS = update_input_faceset(SELECTION_FACES_DATA)
    video_files = prepare_video(Path(args.video_path))
    swap(args, video_files)

def update_input_faceset(face_list):
    input_faceset = []
    for f in face_list:
        face_set = FaceSet()
        face = f[0]
        face.mask_offsets = (0,0)
        face_set.faces.append(face)
        input_faceset.append(face_set)
    return input_faceset



if __name__ == '__main__':
    run()


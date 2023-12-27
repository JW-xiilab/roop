<<<<<<< HEAD
## This project has been discontinued

Yes, you can still use this software. It just won't recieve any updates now.

> The reason behind shutting the project down is that a developer with write access to the code published a [problematic video](https://github.com/s0md3v/roop/commit/cf7ba1caf932e8c9f39d972100f74022e7372c27) to the documentation of the project. This happened while I was taking a break from the project in July-Aug 2023. It went unnoticed for 2 weeks until someone reached out to me to talk about this project. It was a complete breach of trust for me and I decided that I do not have the interest or time to oversee the development of a software with such ethical issues.
> I thank all the amazing people who contributed to this project and made what it is in it's final form.

# Roop

> Take a video and replace the face in it with a face of your choice. You only need one image of the desired face. No dataset, no training.

[![Build Status](https://img.shields.io/github/actions/workflow/status/s0md3v/roop/ci.yml.svg?branch=main)](https://github.com/s0md3v/roop/actions?query=workflow:ci)

## Installation

Be aware, the installation needs technical skills and is not for beginners. Please do not open platform and installation related issues on GitHub. We have a very helpful [Discord](https://discord.com/invite/Y9p4ZQ2sB9) community that will guide you to install roop.

[Basic](https://github.com/s0md3v/roop/wiki/1.-Installation) - It is more likely to work on your computer, but will be quite slow

[Acceleration](https://github.com/s0md3v/roop/wiki/2.-Acceleration) - Unleash the full potential of your CPU and GPU


## Usage

Start the program with arguments:

```
python run.py [options]

-h, --help                                                                 show this help message and exit
-s SOURCE_PATH, --source SOURCE_PATH                                       select an source image
-t TARGET_PATH, --target TARGET_PATH                                       select an target image or video
-o OUTPUT_PATH, --output OUTPUT_PATH                                       select output file or directory
--frame-processor FRAME_PROCESSOR [FRAME_PROCESSOR ...]                    frame processors (choices: face_swapper, face_enhancer, ...)
--keep-fps                                                                 keep target fps
--keep-frames                                                              keep temporary frames
--skip-audio                                                               skip target audio
--many-faces                                                               process every face
--reference-face-position REFERENCE_FACE_POSITION                          position of the reference face
--reference-frame-number REFERENCE_FRAME_NUMBER                            number of the reference frame
--similar-face-distance SIMILAR_FACE_DISTANCE                              face distance used for recognition
--temp-frame-format {jpg,png}                                              image format used for frame extraction
--temp-frame-quality [0-100]                                               image quality used for frame extraction
--output-video-encoder {libx264,libx265,libvpx-vp9,h264_nvenc,hevc_nvenc}  encoder used for the output video
--output-video-quality [0-100]                                             quality used for the output video
--max-memory MAX_MEMORY                                                    maximum amount of RAM in GB
--execution-provider {cpu} [{cpu} ...]                                     available execution provider (choices: cpu, ...)
--execution-threads EXECUTION_THREADS                                      number of execution threads
-v, --version                                                              show program's version number and exit
```


### Headless

Using the `-s/--source`, `-t/--target` and `-o/--output` argument will run the program in headless mode.


## Disclaimer

This software is designed to contribute positively to the AI-generated media industry, assisting artists with tasks like character animation and models for clothing.

We are aware of the potential ethical issues and have implemented measures to prevent the software from being used for inappropriate content, such as nudity.

Users are expected to follow local laws and use the software responsibly. If using real faces, get consent and clearly label deepfakes when sharing. The developers aren't liable for user actions.


## Licenses

Our software uses a lot of third party libraries as well pre-trained models. The users should keep in mind that these third party components have their own license and terms, therefore our license is not being applied.


## Credits

- [deepinsight](https://github.com/deepinsight) for their [insightface](https://github.com/deepinsight/insightface) project which provided a well-made library and models.
- all developers behind the libraries used in this project


## Documentation

Read the [documentation](https://github.com/s0md3v/roop/wiki) for a deep dive.
=======
# roop-unleashed

[Changelog](#changelog) • [Usage](#usage) • [Wiki](https://github.com/C0untFloyd/roop-unleashed/wiki)


Uncensored Deepfakes for images and videos without training and an easy-to-use GUI.


![Screen](https://github.com/C0untFloyd/roop-unleashed/assets/131583554/6ee6860d-efbe-4337-8c62-a67598863637)

### Features

- Platform-independant Browser GUI
- Selection of multiple input/output faces in one go
- Many different swapping modes, first detected, face selections, by gender
- Batch processing of images/videos
- Masking of face occluders using text prompts
- Optional Face Restoration using different enhancers
- Preview swapping from different video frames
- Live Fake Cam using your webcam
- Extras Tab for cutting videos etc.
- Settings - storing configuration for next session
- Theme Support

and lots more...


## Disclaimer

This project is for technical and academic use only.
Users of this software are expected to use this software responsibly while abiding the local law. If a face of a real person is being used, users are suggested to get consent from the concerned person and clearly mention that it is a deepfake when posting content online. Developers of this software will not be responsible for actions of end-users.
**Please do not apply it to illegal and unethical scenarios.**

In the event of violation of the legal and ethical requirements of the user's country or region, this code repository is exempt from liability

### Installation

Please refer to the Wiki.




### Usage

- Windows: run the `windows_run.bat` from the Installer.
- Linux: `python run.py`

<a target="_blank" href="https://colab.research.google.com/github/C0untFloyd/roop-unleashed/blob/main/roop-unleashed.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
  

Additional commandline arguments are currently unsupported and settings should be done via the UI.

> Note: When you run this program for the first time, it will download some models roughly ~2Gb in size.




### Changelog

**16.10.2023** v3.3.4

**11.8.2023** v2.7.0

Initial Gradio Version - old TkInter Version now deprecated

- Re-added unified padding to face enhancers
- Fixed DMDNet for all resolutions
- Selecting target face now automatically switches swapping mode to selected
- GPU providers are correctly set using the GUI (needs restart currently)
- Local output folder can be opened from page
- Unfinished extras functions disabled for now
- Installer checks out specific commit, allowing to go back to first install
- Updated readme for new gradio version
- Updated Colab


# Acknowledgements

Lots of ideas, code or pre-trained models used from the following projects:

https://github.com/deepinsight/insightface
https://github.com/s0md3v/roop
https://github.com/AUTOMATIC1111/stable-diffusion-webui
https://github.com/Hillobar/Rope
https://github.com/janvarev/chain-img-processor
https://github.com/TencentARC/GFPGAN   
https://github.com/kadirnar/codeformer-pip
https://github.com/csxmli2016/DMDNet


Thanks to all developers!

>>>>>>> a55b78d4ec1eb6c48da7d36c1954797959a34f37

import sys
from pathlib import Path

import cv2
import numpy as np
from numpy.linalg import norm as l2norm
from PIL import Image

import torch


# sys.path.append(str(Path(__file__).parents[1]))
from arcface_pytorch.arcface import Arcface


adaface_models = {
    'ir_50':"models/adaface_ir50_webface4m.ckpt",
}

class FaceAnalysis:
    def __init__(self,):
        """
        Adaface -> bounding box, kps, embeddings
        RetinaFace -> bounding box, kps
        Arcface -> embeddings
        Buffalo -> embeddings
        HRNet -> landmarks for rotation check (deprecated)
        """
    
        # self.base = Adaface()
        self.base = Retinaface()
        # self.hrnet = HRNet()

        self.emb = Buffalo().model
        # self.emb = Arcface()
        # self.emb = Adaface()

    
    def get(self, img, max_num=0):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        bbox, det_score, kps = self.base.get_basic(img, pil_img=pil_img)
        if not det_score:
            return []
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        face.embedding = self.emb.get(img, face)
        # face.landmark_2d_106 = self.hrnet.get_lmk(img, face.bbox)
        face.landmark_2d_106 = self.__get_lmk(face.kps)
        return [face]

    def __get_lmk(self, kps):
        forehead = np.mean(kps[0:2], axis=0)
        chin = np.mean(kps[3:], axis=0)
        return {72:forehead, 0:chin}
    

#from easydict import EasyDict

class Face(dict):

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        #for k in self.__class__.__dict__.keys():
        #    if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
        #        setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                    if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            try:
                value = self.__class__(value)
            except TypeError:
                pass

        super(Face, self).__setattr__(name, value)
        super(Face, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        return None

    @property
    def embedding_norm(self):
        if self.embedding is None:
            return None
        return l2norm(self.embedding)

    @property 
    def normed_embedding(self):
        if self.embedding is None:
            return None
        return self.embedding / self.embedding_norm

    @property 
    def sex(self):
        if self.gender is None:
            return None
        return 'M' if self.gender==1 else 'F'

class Adaface():
    def __init__(self,):
        # sys.path.append(str(Path(__file__).parent/'AdaFace'))
        import AdaFace.net as net
        from AdaFace.face_alignment import align
        self.adaface_models = {
            'ir_50':"models/adaface_ir50_webface4m.ckpt",
            }
        self.net = net
        self.align = align
        self.__load_pretrained_model('ir_50')
    
    def __load_pretrained_model(self, architecture='ir_50'):
    # load model and pretrained statedict
        assert architecture in adaface_models.keys()
        self.model = self.net.build_model(architecture)
        statedict = torch.load(str(Path(__file__).parent) + '/' + self.adaface_models[architecture])['state_dict']
        model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
        self.model.load_state_dict(model_statedict)
        self.model.eval()
    
    def get(self, pil_img, face=None):
        aligned_rgb_img = self.align.get_aligned_face(rgb_pil_image=pil_img)
        bgr_tensor_input = self.__to_input(aligned_rgb_img)
        feature, _ = self.model(bgr_tensor_input)
        # face.embedding = feature.detach()
        return feature.detach()

    def get_basic(self, img, pil_img):
        bbox, det_score, kps = self.align.get_aligned_bbox(rgb_pil_image=pil_img)
        return bbox, det_score, kps

    def __to_input(self, pil_rgb_image):
        np_img = np.array(pil_rgb_image)
        brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
        tensor = torch.tensor(brg_img.transpose(2,0,1)).float()
        return tensor.unsqueeze(0)

class HRNet():
    def __init__(self,):
        import HRNet_Facial_Landmark_Detection.lib.models as models
        from HRNet_Facial_Landmark_Detection.lib.utils.transforms import crop_v2, crop
        from HRNet_Facial_Landmark_Detection.lib.config import config, update_config
        from HRNet_Facial_Landmark_Detection.lib.core.evaluation import decode_preds    
        update_config(config, str(self.hrnet_path/'experiments/wflw/face_alignment_wflw_hrnet_w18.yaml'))
        config.defrost()
        config.MODEL.INIT_WEIGHTS = False
        config.freeze()
        self.model = models.get_face_alignment_net(config)
        self.config = config
        self._load()
        self.crop_v2 = crop_v2
        self.decode_preds = decode_preds
    
    def _load(self, model_file='hrnetv2_pretrained/HR18-WFLW.pth'):
        state_dict = torch.load(str(f'HRNet_Facial_Landmark_Detection/{model_file}'))
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _prepare_input(self, image, bbox, image_size):
        """

        :param image:The path of the image to be detected
        :param bbox:The bbox of target face
        :param image_size: refers to config file
        :return:
        """
        scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200
        center_w = (bbox[0] + bbox[2]) / 2
        center_h = (bbox[1] + bbox[3]) / 2
        center = torch.Tensor([center_w, center_h])
        scale *= 1.25
        img = np.float32(image)
        # img = np.array(Image.open(image).convert('RGB'), dtype=np.float32)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = self.crop_v2(img, center, scale, image_size, rot=0)
        img = img.astype(np.float32)
        img = (img / 255.0 - mean) / std
        img = img.transpose([2, 0, 1])
        img = torch.Tensor(img)
        img = img.unsqueeze(0)
        return img, center, scale

    def get_lmk(self, image, bbox):
        input, center, scale = self._prepare_input(image, bbox, self.config.MODEL.IMAGE_SIZE)
        output = self.model(input)
        score_map = output.data.cpu()
        preds = self.decode_preds(score_map, [center], [scale], [64,64])
        return preds.squeeze(0).numpy()

class Buffalo():
    def __init__(self):
        import insightface
        self.model = insightface.model_zoo.get_model('/root/.insightface/models/buffalo_l/w600k_r50.onnx', providers=['CUDAExecutionProvider'])
        self.model.prepare(ctx_id=0)

class Retinaface():
    def __init__(self):
        from pytorch_retinaface.retinaface.pre_trained_models import get_model
        self.model = get_model("resnet50_2020-07-20", max_size=2048, device='cuda')
        self.model.eval()

    def get_basic(self, image, pil_img=None):
        return self.model.predict_jsons(image)

    

if __name__ == "__main__":
    img = cv2.imread('/DATA_17/kjw/01-DeepFake/tmp_dataset/src/my/a_2.png')
    faceanalysis = FaceAnalysis()
    faces = faceanalysis.get(img)
    print(1)


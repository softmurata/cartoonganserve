import torch
from torchvision.utils import save_image
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import os
import boto3
import json
import base64
import io
from PIL import Image

from model import CartoonGAN

from ts.torch_handler.base_handler import BaseHandler

"""
Overwrite BaseHandler class

cannot read model well

"""

class ModelHandler(BaseHandler):
    """[summary]

    Args:
        BaseHandler ([type]): [description]

    Returns:
        [type]: [description]
    """

    def __init__(self):
        self.model = None
        self.device = None
        self.initialized = False
        self.context = None
        self.manifest = None
        self.map_location = None
        self.explain = False
        self.target = 0

        self.style = "Hayao"  # Hosoda, Shinkai, Paprika, Hayao
        self.load_size = 450

        self.image_filename = "target_cargan.png"
        self.bucket_location = "us-east-1"
        self.bucket = "murata-torchserve-db"


    def initialize(self, context):
        """Initialize function loads the model.pt file and initialized the model object.
	    First try to load torchscript else load eager mode state_dict based model.
        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        Raises:
            RuntimeError: Raises the Runtime error when the model.py is missing
        """

        properties = context.system_properties
        self.map_location = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else self.map_location
        )
        self.manifest = context.manifest

        # load model
        # model_pt_path = "/home/ubuntu/murata/Server/cartoonganserve/serve/examples/cartoongan/pretrained_model/{}_net_G_float.pth".format(self.style)  # generator weight
        model_pt_path = "/home/ubuntu/murata/Media2Cloud/Server/cartoonganserve/serve/examples/cartoonganserve/pretrained_model/{}_net_G_float.pth".format(self.style)
        self.model = CartoonGAN()
        self.model.load_state_dict(torch.load(model_pt_path, map_location=self.map_location))
        self.model.to(self.device)
        self.model.eval()

        
        self.initialized = True

    def preprocess(self, data):
        """[summary]
        receive binary data and covert it into image format
        Args:
            data ([type]): [description]

        Returns:
            [type]: [description]
        """

        print("preprocess")

        row = data[0]
        image = row.get("data") or row.get("body")

        if isinstance(image, str):
            image = base64.b64decode(image)
            
        if isinstance(image, (bytearray, bytes)):
            image = Image.open(io.BytesIO(image))
            image = self.process_image(image)
        else:
            image = torch.FloatTensor(image)
            image = torch.stack([image])
        
        return image.to(self.device)

    def process_image(self, image):

        image = image.convert("RGB")

        h = image.size[0]
        w = image.size[1]

        ratio = h * 1.0 / w

        if ratio > 1:
            h = self.load_size
            w = int(h + 1.0 / ratio)
        else:
            w = self.load_size
            h = int(w * ratio)

        image = image.resize((h, w), Image.BICUBIC)
        image = np.asarray(image)

        image = image[:, :, [2, 1, 0]]
        image = transforms.ToTensor()(image).unsqueeze(0)
        image = -1 + 2 * image

        image = Variable(image, volatile=True).to(self.device)
        
        return image

    def inference(self, model_input):
        """[summary]
        model inferences against input
        Args:
            model_input ([type]): [description]

        Returns:
            [type]: [description]
        """
        print("inference")
        model_output = self.model(model_input)
        return model_output
    
    def postprocess(self, inference_output):
        """[summary]
        output for rest api
        basically json
        Args:
            inference_output ([type]): [description]

        Returns:
            [type]: [description]
        """
        postprocess_output = inference_output
        output_image = postprocess_output[0]

        output_image = output_image[[2, 1, 0], :, :]
        output_image = output_image.data.cpu().float() * 0.5 + 0.5
        # convert results into json format
        # file_name = "/home/ubuntu/murata/Server/cartoonganserve/serve/image_dir/{}".format(self.image_filename)
        file_name = "/home/ubuntu/murata/Media2Cloud/Server/cartoonganserve/serve/examples/image_dir/{}".format(self.image_filename)
        save_image(output_image, file_name)

        
        # s3 upload
        # json_format = self.upload_file(file_name, self.bucket, object_name=None)
        url = "https://{}.s3.amazonaws.com/{}".format(self.bucket, self.image_filename)
        json_format = {"url": url}
        json_format = json.dumps(json_format)

        return json_format

    def upload_file(self, file_name, bucket, object_name):
        if object_name is None:
            object_name = file_name

        s3_client = boto3.resource("s3")

        url = "https://{}.s3.amazonaws.com/{}".format(self.bucket, self.image_filename)

        s3_client.Bucket(self.bucket).upload_file(Filename=file_name, Key=self.image_filename)
        
        return json.dumps({"url": url})


# main function
_service = ModelHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return [data]
    except Exception as e:
        raise e
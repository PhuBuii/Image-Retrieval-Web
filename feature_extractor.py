from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L,preprocess_input
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.models import Model
import numpy as np

# See https://keras.io/api/applications/ for details

class VGG16_FE:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def extract(self, img):
        """
        Extract a deep feature from an input image
        Args:
            img: from PIL.Image.open(path) or tensorflow.keras.preprocessing.image.load_img(path)
        Returns:
            feature (np.ndarray): deep feature with the shape=(4096, )
        """
        img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
        return feature / np.linalg.norm(feature)  # Normalize
    
class Xception_FE:
    def __init__(self):
        base_model = Xception(weights='imagenet')  # Create Xception model

        # Create model based on Xception model above
        # With input is the same as Xception model
        # And output results from avg_pool layer of Xception model
        # (None, 299, 299, 3) --> (None, 2048)
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    def extract(self, img):
        img = img.resize((299, 299))  # Image must be 299 x 299 pixels
        img = img.convert('RGB')  # Image must be color photo

        array = image.img_to_array(img)  # Image to np.array with shape (299, 299, 3)
        array = np.expand_dims(array, axis=0)  # (299, 299, 3) --> (1, 299, 299, 3)
        array = preprocess_input(array)  # Subtracting average values for each pixel

        feature = self.model.predict(array)[0]  # Predict with shape (1, 2048) --> (2048, )
        feature = feature / np.linalg.norm(feature)  # Normalize

        return feature

class EfficientNetV2L_FE:
    def __init__(self):
        base_model = EfficientNetV2L(weights='imagenet')  # Create EfficientNetV2L model

        # Create model based on EfficientNetV2L model above
        # With input is the same as EfficientNetV2L model
        # And output results from top_dropout layer of EfficientNetV2L model
        # (None, 480, 480, 3) --> (None, 1280)
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('top_dropout').output)

    def extract(self, img):
        img = img.resize((480, 480))  # Image must be 480 x 480 pixels
        img = img.convert('RGB')  # Image must be color photo

        array = image.img_to_array(img)  # Image to np.array with shape (480, 480, 3)
        array = np.expand_dims(array, axis=0)  # (480, 480, 3) --> (1, 480, 480, 3)
        array = preprocess_input(array)  # Subtracting average values for each pixel

        feature = self.model.predict(array)[0]  # Predict with shape (1, 1280) --> (1280, )
        feature = feature / np.linalg.norm(feature)  # Normalize

        return feature


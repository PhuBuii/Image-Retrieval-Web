from PIL import Image
from feature_extractor import VGG16_FE,Xception_FE,EfficientNetV2L_FE
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    fe_vgg16 = VGG16_FE()
    fe_xception = Xception_FE()
    fe_efficientNetV2L = EfficientNetV2L_FE()

    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg

        feature_vgg16 = fe_vgg16.extract(img=Image.open(img_path))
        feature_path_vgg16 = Path("./static/feature/vgg16") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path_vgg16, feature_vgg16)

        feature_xception = fe_xception.extract(img=Image.open(img_path))
        feature_path_xception = Path("./static/feature/xception") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path_xception, feature_xception)

        feature_efficientNetV2L = fe_efficientNetV2L.extract(img=Image.open(img_path))
        feature_path_efficientNetV2L = Path("./static/feature/efficientNetV2L") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path_efficientNetV2L, feature_efficientNetV2L)
    print("Done ")
    
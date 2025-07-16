import os
import time
import json

import torch
from torchvision import transforms
from model.models_vit_tensor_CD import vit_base_patch16
import skimage.io as io

import numpy as np
from PIL import Image
# from src import UNet
#
# from src import UNet
# import pydensecrf.densecrf as dcrf


# def time_synchronized():
#     torch.cuda.synchronize() if torch.cuda.is_available() else None
#     return time.time()


# def open_image(img_path):
#     # with rasterio.open(img_path) as data:
#     #     img = data.read()  # (c, h, w)
#     img = io.imread(img_path)

#     # return img.transpose(1, 2, 0).astype(np.float32)
#     return img.astype(np.float32)

# def main():
#     palette_path = "/home/tiiairc/GenAI/SpectralGPT/downstream_predict/OSCD/palette.json"

#     # assert os.path.exists(weights_path), f"weights {weights_path} not found."
#     # assert os.path.exists(img_path), f"image {img_path} not found."
#     # assert os.path.exists(palette_path), f"palette {palette_path} not found."
#     with open(palette_path, "rb") as f:
#         pallette_dict = json.load(f)
#         pallette = []
#         for v in pallette_dict.values():
#             pallette += v
#     # get devices
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print("using {} device.".format(device))

#     # create model
#     model = vit_base_patch16()
#     # model = UNet(in_channels=12, num_classes=13, base_c=64)
#     # model = UPerNet(num_classes=13)

#     # delete weights about aux_classifier
#     # weights_dict = torch.load(weights_path, map_location='cpu')['model']
#     # load weights
#     checkpoint = torch.load('/home/tiiairc/GenAI/SpectralGPT/vit49.pth',
#                             map_location=device, weights_only=False)

#     checkpoint_model = {k.replace('module.', ''): v for k, v in checkpoint.items()}
#     # model.load_state_dict(torch.load(weights_path, map_location=device)['model'])
#     model.load_state_dict(checkpoint_model, strict=False)
#     model.to(device)
#     msg = model.load_state_dict(checkpoint_model, strict=False)
#     print(msg)

#     # load image
#     image_folder_path1 = "/OSCD/I1"
#     image_folder_path2 = "/OSCD/I2"#
#     image_folder_path1 = "/home/tiiairc/GenAI/Datasets/Onera Satellite Change Detection dataset - Images/aguasclaras/imgs_1_rect"
#     image_folder_path2 = "/home/tiiairc/GenAI/Datasets/Onera Satellite Change Detection dataset - Images/aguasclaras/imgs_2_rect"#

#     # 获取图片文件夹中的所有图片文件名
#     image_file_names = os.listdir(image_folder_path1)

#     # 遍历图片文件名
#     for image_file_name in image_file_names:
#         img1 = open_image(os.path.join(image_folder_path1, image_file_name))
#         img2 = open_image(os.path.join(image_folder_path2, image_file_name))

#         kid1 = (img1 - img1.min(axis=(0, 1), keepdims=True))
#         mom1 = (img1.max(axis=(0, 1), keepdims=True) - img1.min(axis=(0, 1), keepdims=True))
#         img1 = kid1 / (mom1)

#         kid2 = (img2 - img2.min(axis=(0, 1), keepdims=True))
#         mom2 = (img2.max(axis=(0, 1), keepdims=True) - img2.min(axis=(0, 1), keepdims=True))
#         img2 = kid2 / (mom2)


#             # from pil image to tensor and normalize
#         # data_transform = transforms.Compose([
#         #         transforms.ToTensor(),
#         #     ])
#         data_transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((128, 128)),
#             transforms.ToTensor(),
#         ])
#         img1 = data_transform(img1)
#         img1 = torch.unsqueeze(img1, dim=0)
#         img2 = data_transform(img2)
#         img2 = torch.unsqueeze(img2, dim=0)

#         img1 = img1.cuda()
#         img2 = img2.cuda()
#         print(img1.shape)

#         model.eval()  # 进入验证模式
#         with torch.no_grad():
#                 t_start = time_synchronized()
#                 output = model(img1.to(device),img2.to(device))
#                 t_end = time_synchronized()
#                 print("inference time: {}".format(t_end - t_start))

#                 prediction = output.argmax(1).squeeze(0)
#                 prediction = prediction.to("cpu").numpy().astype(np.uint8)
#                 # print(prediction)

#                 mask = Image.fromarray(prediction)
#                 mask.putpalette(pallette)
#                 mask.save(os.path.join("/home/tiiairc/GenAI/SpectralGPT/downstream_tasks/OSCD/res/", image_file_name))



# if __name__ == '__main__':
#     main()


import os
import time
import json
import torch
from torchvision import transforms
from model.models_vit_tensor_CD import vit_base_patch16
import numpy as np
from PIL import Image
from skimage import io
from scipy.ndimage import zoom

# === Constants ===
NORMALISE_IMGS = True
TARGET_SIZE = (128, 128)

# === Sentinel-2 band reader for TYPE 4 ===
def adjust_shape(I, s):
    I = I[:s[0], :s[1]]
    si = I.shape
    p0 = max(0, s[0] - si[0])
    p1 = max(0, s[1] - si[1])
    return np.pad(I, ((0, p0), (0, p1)), 'edge')

def read_sentinel_img_band12(path):
    im_name = os.listdir(path)[0][:-7]

    r = io.imread(path + im_name + "B04.tif")
    s = r.shape
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")
    nir = io.imread(path + im_name + "B08.tif")

    ir1 = adjust_shape(zoom(io.imread(path + im_name + "B05.tif"), 2), s)
    ir2 = adjust_shape(zoom(io.imread(path + im_name + "B06.tif"), 2), s)
    ir3 = adjust_shape(zoom(io.imread(path + im_name + "B07.tif"), 2), s)
    nir2 = adjust_shape(zoom(io.imread(path + im_name + "B8A.tif"), 2), s)
    swir2 = adjust_shape(zoom(io.imread(path + im_name + "B11.tif"), 2), s)
    swir3 = adjust_shape(zoom(io.imread(path + im_name + "B12.tif"), 2), s)

    uv = adjust_shape(zoom(io.imread(path + im_name + "B01.tif"), 6), s)
    wv = adjust_shape(zoom(io.imread(path + im_name + "B09.tif"), 6), s)

    I = np.stack((uv, b, g, r, ir1, ir2, ir3, nir, nir2, wv, swir2, swir3), axis=2).astype('float')
    return I  # shape: [H, W, 12]

def preprocess_image(img):
    # Normalize per channel using min-max
    img = img.transpose(2, 0, 1)  # [C, H, W]
    kid = (img - img.min(axis=(1, 2), keepdims=True))
    mom = (img.max(axis=(1, 2), keepdims=True) - img.min(axis=(1, 2), keepdims=True))
    img = kid / (mom + 1e-6)

    # Resize using bilinear interpolation
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1, C, H, W]
    img = torch.nn.functional.interpolate(img, size=TARGET_SIZE, mode='bilinear', align_corners=False)
    return img  # [1, 12, 128, 128]

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def main():
    palette_path = "/home/tiiairc/GenAI/SpectralGPT/downstream_predict/OSCD/palette.json"
    with open(palette_path, "rb") as f:
        pallette = sum(json.load(f).values(), [])  # Flatten palette

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = vit_base_patch16()
    ckpt = torch.load('/home/tiiairc/GenAI/SpectralGPT/change_default/vit58.pth', map_location=device, weights_only=False)
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()

    path_base = "/home/tiiairc/GenAI/Datasets/Onera Satellite Change Detection dataset - Images/brasilia"
    folder1 = os.path.join(path_base, "imgs_1_rect/")
    folder2 = os.path.join(path_base, "imgs_2_rect/")
    out_dir = "/home/tiiairc/GenAI/SpectralGPT/downstream_tasks/OSCD/res"
    os.makedirs(out_dir, exist_ok=True)

    image_names = os.listdir(folder1)

    for name in image_names:
        # print(name)
        scene_name = name[:-7]  # strip off band suffix like B04.tif
        path1 = os.path.join(folder1, scene_name + "B04.tif")  # anchor band to check presence
        if not os.path.exists(path1):
            continue

        I1 = read_sentinel_img_band12(folder1)
        I2 = read_sentinel_img_band12(folder2)

        img1 = preprocess_image(I1).to(device)
        img2 = preprocess_image(I2).to(device)

        with torch.no_grad():
            t_start = time_synchronized()
            print(img1.shape)
            output = model(img1, img2)
            t_end = time_synchronized()
            print(f"Inference time for {scene_name}: {t_end - t_start:.3f}s")

            prediction = output.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            # print(out_dir)
            mask = Image.fromarray(prediction)
            
            mask.putpalette(pallette)
            mask.save(os.path.join(out_dir, "brasilia" + ".png"))
            break

if __name__ == "__main__":
    main()

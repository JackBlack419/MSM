import torch
import clip
import os
import sys
import h5py
from PIL import Image
from tqdm import tqdm

# 获取当前文件所在的绝对路径
current_path = os.path.abspath(__file__)

# 获取当前文件所在目录的父目录的绝对路径
parent_dir = os.path.dirname(os.path.dirname(current_path))

# 将父目录添加到sys.path中
sys.path.append(parent_dir)

# 现在可以从A目录中导入config.py
from config import Config
opt = Config()

MSRVTT_frame_path = opt.MSRVTT_frame_path # 存储了MSVD视频帧的path
MSRVTT_feature_path = opt.MSRVTT_features_path # 存放MSVD经CLIP的preprocess处理后的视频特征的path
MSVD_frame_path = opt.MSVD_frame_path # 存放了MSRVTT视频帧的path
MSVD_feature_path = opt.MSVD_features_path # 存放MSRVTT经CLIP的preprocess处理后的视频特征的path
CLIP_weights_path = opt.CLIP_weights_path # 存放了CLIPweights的路径

# print(clip.available_models()) #['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_VIT, preprocess_VIT = clip.load(name=os.path.join(CLIP_weights_path,'ViT-B-32.pt'), device=device)
model_RN101, preprocess_RN101 = clip.load(name=os.path.join(CLIP_weights_path,'RN101.pt'), device=device)



def image_encode(model,image_input):
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    return image_features.cpu()
# def text_encode(model,text_input):
#     with torch.no_grad():
#         token = clip.tokenize(text_input).to(device)
#         text_features = model.encode_text(token)
#     return text_features


def extract_image_feature(frame_path, model, preprocess, dataset_name, backbone):
    # print(model)
    video_feature_save_path = os.path.join(opt.video_feature_path,f'{dataset_name}.h5')
    with h5py.File(video_feature_save_path,'a') as h5_file:
        if backbone not in h5_file:
            captions_grp = h5_file.create_group(backbone)
        else:
            captions_grp = h5_file[backbone]  # 如果组已经存在，则获取该组
            
        for item in os.listdir(frame_path):  # 遍历train,test,valid三个文件夹下的文件
            sub_grp = captions_grp.create_group(item)
            root_path = os.path.join(frame_path, item)
            for video_id in tqdm(os.listdir(root_path), desc='dirs_processing'):  # 遍历train或test或valid数据集下的图片文件夹
                video_path = os.path.join(root_path, video_id)
                iamge_input_list = []
                for pic in os.listdir(video_path):  # 遍历视频中每一张图片
                    pic_path = os.path.join(video_path, pic)
                    pic_input = Image.open(pic_path)
                    image_input = preprocess(pic_input)  # 处理每一张图片
                    iamge_input_list.append(image_input)  # 添加每一张图片的数据
                # all_image_input = torch.stack(iamge_input_list)  # 把20帧数据汇总
                all_image_input = torch.stack(iamge_input_list).to(device)  # 把20帧数据汇总 torch.Size([20, 512])
                image_feature = image_encode(model, all_image_input)  # 对20帧图片数据一次性编码 类型为torch.float16
                sub_grp.create_dataset(video_id, data=image_feature)

extract_image_feature(MSRVTT_frame_path, model_RN101, preprocess_RN101, 'MSRVTT','RN101')
extract_image_feature(MSRVTT_frame_path, model_VIT, preprocess_VIT, 'MSRVTT','VIT')
extract_image_feature(MSVD_frame_path, model_RN101, preprocess_RN101, 'MSVD','RN101')
extract_image_feature(MSVD_frame_path, model_VIT, preprocess_VIT, 'MSVD','VIT')
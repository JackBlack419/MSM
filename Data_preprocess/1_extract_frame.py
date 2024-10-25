import cv2
import sys
import os
import json
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# 获取当前文件所在的绝对路径
current_path = os.path.abspath(__file__)

# 获取当前文件所在目录的父目录的绝对路径
parent_dir = os.path.dirname(os.path.dirname(current_path))

# 将父目录添加到sys.path中
sys.path.append(parent_dir)

# 现在可以从A目录中导入config.py
from config import Config
opt = Config()

captions_path = opt.captions_path # 存放captions的路劲
MSRVTT_video_path = opt.MSRVTT_video_path # 存放MSRVTT视频的路径
MSRVTT_frame_path = opt.MSRVTT_frame_path # 存放MSRVTT提取出的视频帧的路径
MSVD_video_path = opt.MSVD_video_path # 存放MSVD视频的路径
MSVD_frame_path = opt.MSVD_frame_path # 存放提取出的MSVD视频帧的路径

# 在存放MSVD和MSRVTT的视频帧的文件夹下划分数据集为train、test、valid
folders = ['train', 'test', 'valid']
for folder in folders:
    os.makedirs(os.path.join(MSRVTT_frame_path,folder),exist_ok=True)
    os.makedirs(os.path.join(MSVD_frame_path,folder),exist_ok=True)
    

def process_frame(idx, video_path, store_path, item_name):
    """
    处理单帧的函数，用于多线程执行。
    """
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            print(f"Frame at index {idx} could not be read.")
            cap.release()
            return None

        # frame = cv2.resize(frame, (224, 224))

        # 保存帧为图像文件
        os.makedirs(os.path.join(store_path, f"{item_name}"), exist_ok=True)
        frame_path = os.path.join(store_path, f"{item_name}")

        cv2.imwrite(os.path.join(frame_path, f'frame_{idx}.jpg'), frame)

        cap.release()
        return frame_path

    except Exception as e:
        print(f"Error processing frame {idx}: {e}")
        return None

def extract_frames_multithread(video_path, store_path, item_name, num_workers=16):
    """
    使用多线程提取和处理视频帧，并保存为图像文件。
    """
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = [int(i) for i in np.linspace(0, total_frames - 1, 20)]  # 每个视频提取20帧的图片
        cap.release()  # 关闭原始的cap，因为在多线程中不需要保持打开状态

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 使用线程池提交任务
            futures = {executor.submit(process_frame, idx, video_path, store_path, item_name): idx for idx in sample_indices}

            frame_paths = []
            for future in futures:
                frame_path = future.result()
                if frame_path is not None:
                    frame_paths.append(frame_path)

        if frame_paths:  # 确保有帧被成功处理
            print("Frames saved successfully.")
        else:
            print("No frames were processed successfully.")

    except Exception as e:
        print(f"Error extracting frames: {e}")


# 从MSRVTT或MSVD的captions中分别取出train、test、valid中每个视频的ID，让我们在保存视频帧的时候能正确保存在对应train、test、valid的目录下
def get_video_id(path):
    video_id_key = []
    with open(path,'r') as file:
        dict = json.load(file)['captions']
    for item in dict.keys():
        if item not in video_id_key:
            video_id_key.append(item)
    return video_id_key


train_video_id = get_video_id(os.path.join(opt.captions_path, 'MSRVTT_train_captions.json'))
test_video_id = get_video_id(os.path.join(opt.captions_path, 'MSRVTT_test_captions.json'))
valid_video_id = get_video_id(os.path.join(opt.captions_path, 'MSRVTT_valid_captions.json'))

for filename in tqdm(os.listdir(MSRVTT_video_path), desc='Processing videos'):
    if filename.endswith('.mp4') or filename.endswith('.avi'):
        file_name = filename.split('.')[0]
        if file_name in train_video_id:
            extract_frames_multithread(os.path.join(MSRVTT_video_path,filename),os.path.join(MSRVTT_frame_path,'train'),file_name)
        elif file_name in valid_video_id:
            extract_frames_multithread(os.path.join(MSRVTT_video_path,filename),os.path.join(MSRVTT_frame_path,'valid'),file_name)
        else:
            extract_frames_multithread(os.path.join(MSRVTT_video_path,filename),os.path.join(MSRVTT_frame_path,'test'),file_name)

# 切分MSVD的数据集
train_video_id = get_video_id(os.path.join(opt.captions_path, 'MSVD_train_captions.json'))
test_video_id = get_video_id(os.path.join(opt.captions_path, 'MSVD_test_captions.json'))
valid_video_id = get_video_id(os.path.join(opt.captions_path, 'MSVD_valid_captions.json'))

for filename in tqdm(os.listdir(MSVD_video_path), desc='Processing videos'):
    if filename.endswith('.mp4') or filename.endswith('.avi'):
        file_name = filename.split('.')[0]
        if file_name in train_video_id:
            extract_frames_multithread(os.path.join(MSVD_video_path,filename),os.path.join(MSVD_frame_path,'train'),file_name)
        elif file_name in valid_video_id:
            extract_frames_multithread(os.path.join(MSVD_video_path,filename),os.path.join(MSVD_frame_path,'valid'),file_name)
        else:
            extract_frames_multithread(os.path.join(MSVD_video_path,filename),os.path.join(MSVD_frame_path,'test'),file_name)
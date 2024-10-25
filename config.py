import argparse  # 导入argparse模块

class Config:
    raw_captions_path = '../captions/raw_captions'
    captions_path = '../captions' # 存放了captions的文件夹
    CLIP_weights_path = '../CLIP/CLIP_weights'
    MSRVTT_video_path = "/media/disk6t/MSR-VTT/data/train-video/train-video" #MSRVTT的视频文件夹
    MSRVTT_frame_path = '../MSRVTT_frame' # MSRVTT存放提取出来的20帧图片的文件夹
    MSRVTT_features_path = "../video_feature/MSRVTT.h5" # MSRVTT存放由CLIP提取出来的视频特征的h5文件的路径
    MSVD_video_path = "/media/disk6t/MSVD/YouTubeClips" # MSVD的视频文件夹
    MSVD_frame_path = "../MSVD_frames" # MSVD存放提取出来的20帧图片的文件夹
    MSVD_features_path= '../video_feature/MSVD.h5' # MSVD存放由CLIP提取出来的视频特征的h5文件的路径
    video_feature_path = '../video_feature'
    BertTokenizer_path = '../bert_base_uncased' # 存放了bert的配置文件的路径
    # CLIP4clip_weights_path = '../CLIP4CLIP' # 存放了CLIP4clip模型的权重参数的路径
    vocab_size = 30522
    log_dir = '../logs' # 保存logs的文件路径
    max_len = 20
    batch_size = 256
    lr = 1e-4
    # d_model = 512
    d_ff = 2048
    n_layers = 6
    heads = 8
    dropout_rate = 0.1
    MSRVTT_num_epochs = 20
    MSVD_num_epochs = 25
    PAD_ID = 0
    model_weights_path = '../model_weights'
    test_and_inference_res = '../inference_res'

# 用来装载参数的容器
parser = argparse.ArgumentParser(description='hypeparameter of model')
# 给这个解析对象添加命令行参数
parser.add_argument('backbone', type=str, help='choose backbone')
parser.add_argument('d_model', type=int, help='choose features')
# parser.add_argument('MSRVTT_features_path', type=str, help='choose features')
# parser.add_argument('MSVD_features_path', type=str, help='choose features')
# parser.add_argument('loss_weight_1', type=float, help='first loss weight')
# parser.add_argument('loss_weight_2', type=float, default=0, help='second loss weight')
# parser.add_argument('loss_weight_3', type=float, default=0, help='third loss weight')
# parser.add_argument('pattern', type=str, help='loss pattern')
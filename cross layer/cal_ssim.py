
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2

def calculate_ssim(test_video_path, reference_video_path):
    # 打开视频文件
    test_video = cv2.VideoCapture(test_video_path)
    reference_video = cv2.VideoCapture(reference_video_path)

    # 读取视频的第一帧
    _, test_frame = test_video.read()
    _, reference_frame = reference_video.read()

    # 确保两帧是相同的尺寸
    try:
        test_frame = cv2.resize(test_frame, (reference_frame.shape[1], reference_frame.shape[0]))
        # print(test_frame.shape)
        # print(reference_frame.shape)
        # 计算SSIM
        ssim_value = ssim(test_frame, reference_frame, channel_axis = -1)



    except cv2.error:
        ssim_value = -1

    # 释放视频文件
    test_video.release()
    reference_video.release()
    return ssim_value



def calculate_psnr(test_video_path, reference_video_path):
    # 打开视频文件
    test_video = cv2.VideoCapture(test_video_path)
    reference_video = cv2.VideoCapture(reference_video_path)

    # 读取视频的第一帧
    _, test_frame = test_video.read()
    _, reference_frame = reference_video.read()

    # 确保两帧是相同的尺寸
    try:
        test_frame = cv2.resize(test_frame, (reference_frame.shape[1], reference_frame.shape[0]))
        # print(test_frame.shape)
        # print(reference_frame.shape)
        # 计算PSNR
        psnr_value = psnr(test_frame, reference_frame)

    except cv2.error:
        psnr_value = 0

    # 释放视频文件
    test_video.release()
    reference_video.release()
    return psnr_value


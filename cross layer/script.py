# assess the ssim of all transmitted video streams

from cal_ssim import calculate_ssim
import os

reference_video_path = 'C:/Users/liuza/Desktop/cross layer/video_ref'
test_video_path = 'C:/Users/liuza/Desktop/cross layer/temp/video_out_hyb'
ssim = []
minus1 = []


for test_video in os.listdir(test_video_path):
    ref_video = reference_video_path + '/' + test_video
    print(ref_video)
    te_video = test_video_path + '/' + test_video
    a = calculate_ssim(te_video, ref_video)
    if a == -1:
        minus1.append(-1)
    else:
        ssim.append(a)

print(len(minus1))
print("Average SSIM:",sum(ssim)/len(ssim))
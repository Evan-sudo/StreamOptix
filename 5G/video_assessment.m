%% This file is written for extracting objective assessments of the videos
% SSIM and PSNR of the video is calculated
videopath = '/Users/evan/Desktop/studies & research/final year project/video_out/'; 
video_files = dir([videopath '*.mp4']);


psnr = string(zeros(length(video_files),2));
ssim = string(zeros(length(video_files),2));

ref_vid = '/Users/evan/Desktop/studies & research/final year project/videos_representations/BigBuckBunny/1920x1080_fps30_420_7000k.mp4';   % Target video


for ind = 1:length(video_files)
    cpm_video = [videopath video_files(ind).name];
    try
        psnr(ind,1) = video_files(ind).name;
        ssim(ind,1) = video_files(ind).name;
        psnr(ind,2) = num2str(get_psnr(cpm_video, ref_vid));
        ssim(ind,2) = num2str(get_ssim(cpm_video, ref_vid));
    catch
        psnr(ind,1) = video_files(ind).name;
        ssim(ind,1) = video_files(ind).name;
        psnr(ind,2) = '0'; 
        ssim(ind,2) = '0';
    end
end



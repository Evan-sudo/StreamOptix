function avg_simi = get_ssim (cpr_vid, ref_vid)
comp_video = VideoReader(cpr_vid);
refer_video = VideoReader(ref_vid);
comp_video_seq = read(comp_video,[1,Inf]);
refer_video_seq = read(refer_video,[1,Inf]);
seq_size = size(refer_video_seq);
seq_length = seq_size(4);
imsize = seq_size(1:2);
simi_seq = [];
    for ii = 1:seq_length
        cpr_img = squeeze(comp_video_seq(:,:,:,ii));
        ref_img = squeeze(refer_video_seq(:,:,:,ii));
        cpr = imresize(cpr_img,imsize);
        simi = ssim(cpr, ref_img);
        simi_seq = [simi_seq, simi];
    end
    avg_simi = mean(simi_seq);
end
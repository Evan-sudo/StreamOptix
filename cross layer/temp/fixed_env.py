import numpy as np
import matlab.engine
import os
from cal_ssim import *
from v2b import *

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
VIDEO_CHUNCK_LEN = 2000.0  # millisec, every time add this amount to buffer
BITRATE_LEVELS = 6
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
LINK_RTT = 80  # millisec
eng = matlab.engine.start_matlab()
eng.cd(r'C:\\Users\\liuza\\Desktop\\5g pdsch', nargout=0)
carrier, pdsch, channel = eng.init(nargout=3)
video_list = os.listdir(r'C:\\Users\\liuza\\Desktop\\cross layer\\streaming\\video1')  # read the video list
TOTAL_VIDEO_CHUNCK = len(video_list)
SSIM_cal_en = True
STREAM_PATH = 'C:Users\\liuza\\Desktop\\cross layer\\video\\'  # Path of the transmitted video stream
reference_video_path = 'C:\\Users\\liuza\\Desktop\\cross layer\\video_ref\\'



class Environment:
    def __init__(self, pdsch, channel, carrier, eng):
        
        self.channel = channel
        self.carrier = carrier
        self.pdsch = pdsch
        self.eng = eng
        self.eng.cd(r'C:\\Users\\liuza\\Desktop\\5g pdsch', nargout=0)
        self.video_chunk_counter = 0
        self.buffer_size = 0


    def get_video_chunk(self, quality, retrans):

        assert quality >= 0
        assert quality < BITRATE_LEVELS

        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms

        video_path = 'C:\\Users\\liuza\\Desktop\\cross layer\\streaming\\' + f'video{quality}' + '\\' + video_list[self.video_chunk_counter]
        video_chunk_size = os.path.getsize(video_path)/16   # get video size from the txt file, divided by 2*8 (byte and space) unit: Byte
        print(video_chunk_size)
        print(self.eng)
        delay, ber, BLER, thr = self.eng.trans_vid_seg(video_path, self.pdsch, self.channel, self.carrier, retrans, nargout=4)
        delay += LINK_RTT
        thr = np.array(thr)
        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNCK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time 


        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size
        end_of_video = False
        self.video_chunk_counter += 1
        video_chunk_remain = TOTAL_VIDEO_CHUNCK - self.video_chunk_counter
        if self.video_chunk_counter == len(video_list):
            end_of_video = True


        if SSIM_cal_en:
            bin2hex(STREAM_PATH + video_list[self.video_chunk_counter], 'outhex.txt')
            with open('outhex.txt', 'r', encoding='utf-8') as f:    
                file = f.read()
                file = bytes.fromhex(file)
                with open(STREAM_PATH + video_list[self.video_chunk_counter].split('.') + '.mp4', 'wb') as f2:
                    f2.write(file)
            refer_video = reference_video_path + video_list[self.video_chunk_counter].split(".")[0] + '.mp4'
            ssim = calculate_ssim(STREAM_PATH + video_list[self.video_chunk_counter].split('.') + '.mp4', refer_video)
            if ssim == -1:
                retrans = True
            # if decoding error, video_counter -=1
            

        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            end_of_video, \
            video_chunk_remain, \
            BLER, \
            ber, retrans, ssim
    
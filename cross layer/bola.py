import numpy as np
import fixed_env as env
import matlab.engine
import matplotlib.pyplot as plt
import itertools
import os
import math


S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
CHUNK_LEN = 4
SLEEP_TIME = 500 # in milliseconds 
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [375,1050,1750,3000,4300,5800]  # Kbps
#BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
REBUF_PENALTY = 10.3  # sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 3  # default video quality without agent
LOG_FILE = './results/log_sim_bola'
V = 0.93 
rp = 5
video_list = os.listdir(r'C:\\Users\\liuza\\Desktop\\cross layer\\streaming\\video1')
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = './models/nn_model_ep_5900.ckpt'
streaming_path = r'C:\\Users\\liuza\\Desktop\\cross layer\\streaming'
TOTAL_VIDEO_CHUNKS = len(video_list)-1




def main():

    eng = matlab.engine.start_matlab()
    eng.cd(r'C:\\Users\\liuza\\Desktop\\5g pdsch', nargout=0)
    carrier, pdsch, channel = eng.init(nargout=3)
    print(eng)

    net_env = env.Environment(pdsch, channel, carrier, eng)

    log_path = LOG_FILE
    log_file = open(log_path, 'w')
 
    time_stamp = 0


    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY
    retrans = False


    r_batch = []

    while True:  # serve video forever
        # reward is video quality - rebuffer penalty
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, end_of_video, \
        video_chunk_remain, BLER, ber, retrans, ssim, psnr,  bandwidth = \
            net_env.get_video_chunk(bit_rate, retrans)
        print(delay)
        utility =  [math.log(x/max(VIDEO_BIT_RATE)) for x in VIDEO_BIT_RATE]

        while buffer_size/CHUNK_LEN > V*(max(utility)+rp):  # if not condition, pause requesting new chunk till buffer size is small enough
            sleep_time = sleep_time + SLEEP_TIME
            buffer_size = buffer_size - SLEEP_TIME/1000
    

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # log scale reward
        # log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[0]))
        # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))
                # reward is video quality - rebuffer penalty
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                           VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        # reward = log_bit_rate \
        #          - REBUF_PENALTY * rebuf \
        #          - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

        # reward = BITRATE_REWARD[bit_rate] \
        #          - 8 * rebuf - np.abs(BITRATE_REWARD[bit_rate] - BITRATE_REWARD[last_bit_rate])
        

        last_bit_rate = bit_rate
        bandwidth = (video_chunk_size*8/1000) / (delay)

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(str("{:.3f}".format(time_stamp / M_IN_K)) + '\t' +
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str("{:.3f}".format(buffer_size)) + '\t' +
                       str("{:.3f}".format(rebuf)) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str("{:.3f}".format(delay)) + '\t' +
                       str("{:.3f}".format(BLER)) + '\t' +
                       str("{:.5f}".format(ber)) + '\t' +
                       str("{:.3f}".format(bandwidth)) + ' '+ 'Mbps' + '\t' +
                       str("{:.3f}".format(ssim))  + '\t' +
                       str("{:.3f}".format(psnr))  + '\t' +
                       str("{:.3f}".format(retrans))  + '\t' +
                       str("{:.3f}".format(reward)) + '\n')
        log_file.flush()       
        
        
        # made bola optimal decision
        candidates = []
        for i in range(6):
            bola_re = (V*(utility[i]+rp)-buffer_size/CHUNK_LEN)/(VIDEO_BIT_RATE[i]*M_IN_K*CHUNK_LEN)
            candidates.append(bola_re)
        bit_rate = candidates.index(max(candidates))
        if retrans:
            bit_rate = 0
        

        r_batch.append(reward)

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here

            del r_batch[:]



if __name__ == '__main__':
    main()
import numpy as np
import fixed_env as env
import matlab.engine


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
VIDEO_BIT_RATE = [375,1050,1750,3000,4300,5800]  # Kbps
M_IN_K = 1000.0
REBUF_PENALTY = 10.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RAND_RANGE = 1000000
RESEVOIR = 5  # BB
CUSHION = 10  # BB
LOG_FILE = './results/log_sim_bb'

# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward


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

    r_batch = []
    th = np.array([])
    retrans = False


    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, end_of_video, \
        video_chunk_remain, BLER, ber, retrans, ssim, psnr, bandwidths = \
            net_env.get_video_chunk(bit_rate, retrans)
        print(delay)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                           VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        r_batch.append(reward)

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

        if buffer_size < RESEVOIR:
            bit_rate = 0
        elif buffer_size >= RESEVOIR + CUSHION:
            bit_rate = A_DIM - 1
        else:
            bit_rate = (A_DIM - 1) * (buffer_size - RESEVOIR) / float(CUSHION)

        bit_rate = int(bit_rate)
        if retrans:
            bit_rate = 0

        if end_of_video:
            log_file.write('\n')
            log_file.close()
            break


if __name__ == '__main__':
    main()
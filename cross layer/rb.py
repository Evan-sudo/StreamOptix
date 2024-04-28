import numpy as np
import fixed_env as env
import matlab.engine
import matplotlib.pyplot as plt
import itertools
import os


S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
MPC_FUTURE_CHUNK_COUNT = 4
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [375,1050,1750,3000,4300,5800]  # Kbps
#BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
REBUF_PENALTY = 10.3  # sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 3  # default video quality without agent
LOG_FILE = './results/log_sim_rb'
video_list = os.listdir(r'C:\\Users\\liuza\\Desktop\\cross layer\\streaming\\video1')
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = './models/nn_model_ep_5900.ckpt'
streaming_path = r'C:\\Users\\liuza\\Desktop\\cross layer\\streaming'
TOTAL_VIDEO_CHUNKS = len(video_list)-1


# past errors in bandwidth
past_errors = []
past_bandwidth_ests = []


def update_list(existing, new):
    existing = np.roll(existing, -len(new))
    existing[-len(new):] = new
    return existing



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

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    retrans = False
    past_bandwidths  = np.zeros(5)

   


    while True:  # serve video forever
        # reward is video quality - rebuffer penalty
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, end_of_video, \
        video_chunk_remain, BLER, ber, retrans, ssim, psnr, bandwidths = \
            net_env.get_video_chunk(bit_rate, retrans)
        print(delay)

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

        # retrieve previous state
        if len(s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = rebuf
        #state[3, -1] = float(video_chunk_size)*8 / float(delay)  # kbps
        #state[4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        # state[5: 10, :] = future_chunk_sizes / M_IN_K / M_IN_K

        # ================== Rate Based =========================
        # curr_error = 0 # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
        # if ( len(past_bandwidth_ests) > 0 ):
        #     curr_error  = abs(past_bandwidth_ests[-1]-state[3,-1])/float(state[3,-1])
        # past_errors.append(curr_error)

        # pick bitrate according to MPC           
        # first get harmonic mean of last 5 bandwidths
        if len(bandwidths) == 0:
            pass
        else:
            if len(bandwidths) > len(past_bandwidths):
                past_bandwidths = update_list(past_bandwidths, bandwidths[-len(past_bandwidths):])
            else:
                past_bandwidths = update_list(past_bandwidths, bandwidths)

        # past_bandwidths = state[3,-5:]
        # while past_bandwidths[0] == 0.0:
        #     past_bandwidths = past_bandwidths[1:]
        #if ( len(state) < 5 ):
        #    past_bandwidths = state[3,-len(state):]
        #else:
        #    past_bandwidths = state[3,-5:]
                


        # multi-step prediction  
        prediction = []
        BW_history = past_bandwidths
        while len(prediction) < 4:
            length = 0
            bandwidth_sum = 0
            for past_val in BW_history:
                if past_val != 0:
                    bandwidth_sum += (1/float(past_val))
                    length+=1
            harmonic_bandwidth = 1.0/(bandwidth_sum/length)
            print('Past: ',BW_history)
            print(harmonic_bandwidth)
            prediction.append(harmonic_bandwidth)
            BW_history = np.roll(BW_history,-1)
            BW_history[-1] = harmonic_bandwidth
        future_bandwidth = sum(prediction)/len(prediction)   # unit: Bps
        print('future bandwidth:', future_bandwidth)


        # future bandwidth prediction
        # divide by 1 + max of last 5 (or up to 5) errors
        # max_error = 0
        # error_pos = -5
        # if ( len(past_errors) < 5 ):
        #     error_pos = -len(past_errors)
        # max_error = float(max(past_errors[error_pos:]))
        # future_bandwidth = harmonic_bandwidth/(1+max_error)  # robustMPC here
        # past_bandwidth_ests.append(harmonic_bandwidth)

        bit_rate = min(range(len(VIDEO_BIT_RATE)),key = lambda i: abs(VIDEO_BIT_RATE[i]*M_IN_K/8-future_bandwidth))
        if retrans:
            bit_rate = 0
        s_batch.append(state)

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)


if __name__ == '__main__':
    main()
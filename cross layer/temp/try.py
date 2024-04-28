import numpy as np

MILLISECONDS_IN_SECOND = 1000
BITS_IN_BYTE = 8
B_IN_MB = 1000000
trace = np.load("pose.npy")*B_IN_MB/8
print(trace[0:20])
video_chunk_size = 1960829 # in B


# compare with channel capacity trace
size = 0  # bit?
delay_0 = 0  # ms
timeline = 0
ind = int(timeline // 500)   # integer
if timeline % 500 == 0:
    bandwidth_start = ind
else: bandwidth_start = ind+1  # start of bandwidth prediction
portion = timeline - ind*500 # in ms
print(portion)
size+=portion/MILLISECONDS_IN_SECOND*trace[ind] # in byte
ind+=1
if size > video_chunk_size:
    delay_0 = video_chunk_size/trace[ind]*MILLISECONDS_IN_SECOND
else:
    delay_0 += size/trace[ind]*MILLISECONDS_IN_SECOND
    while (1):
        if size + 0.5*trace[ind]> video_chunk_size:
            delay_0 += (video_chunk_size - size)/trace[ind]*MILLISECONDS_IN_SECOND
            break
        size += 0.5*trace[ind]
        delay_0 += 500
        ind += 1
print('delay 0: ',delay_0)

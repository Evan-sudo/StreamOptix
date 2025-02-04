import re
import os, sys


#video_out_bin = 'C:/Users/liuza/Desktop/cross layer/video'
video_out = 'C:/Users/liuza/Desktop/cross layer/video_out'
video_out_bin = 'C:/Users/liuza/Desktop/cross layer/streaming/video5'

#二进制读取video文件然后转十六进制，输出out.txt
def bi_read_video (your_video_path, your_outfile_path):
    f = open(your_video_path, 'rb')
    outfile1 = open(your_outfile_path, "w")
    i = 0
    while 1:
        c = f.read(1)
        i = i + 1
        if not c:
            break
        #if i % 33 == 0:
        #   outfile1.write("\n")
        '''
        else:
            if ord(c) <= 15:
                outfile1.write(("0x0" + hex(ord(c))[2:])[2:])
            else:
                outfile1.write((hex(ord(c)))[2:])
        '''
        c = (bin(ord(c)))[2:]
        if len(c) < 8:
            c = (8 - len(c))*'0'+c
        c = re.findall(".{1}",c)
        c = " "+" ".join(c)
        outfile1.write(c)
        
        
    outfile1.close()
    f.close()


# def bin2hex(your_video_path, your_outfile_path):
#     with open(your_video_path,'r', encoding='utf-8') as f:
#         f2 = open(your_outfile_path,'w')
#         while True:
#             read = f.read(8)  # 每次读取8个字符，即一个字节的二进制数据
#             if not read:
#                 break
#             read = read.replace(" ", "")  # 去除空格
#             c = hex(int(read, 2))[2:]
#             f2.write(c)
#         f2.close()
    

def bin2hex(your_video_path, your_outfile_path):
    with open(your_video_path,'r', encoding='utf-8') as f:
        f2 = open (your_outfile_path,'w')
        while (True):
            read = f.read(4) 
            if not read or read == '\n':
                break
            c = hex(int(read,2))[2:]
            f2.write(c)
        f2.close()
        

if __name__ == "__main__":
    #out = bi_read_video('320x240_fps30_420_235k.mp4','out.txt') # binary
    for filename in os.listdir(video_out_bin):
        if filename != '.DS_Store' and filename != 'log.txt':
            file_name = os.path.splitext(filename)[0]
            bin2hex(video_out_bin+'/'+filename, 'outhex.txt')
            with open('outhex.txt', 'r', encoding='utf-8') as f:    
                file = f.read()
                file = bytes.fromhex(file)
                with open(video_out+'/'+file_name+'.mp4', 'wb') as f2:
                    f2.write(file)




# if __name__ == '__main__':
#     isExist = os.path.exists(output_path)
#     if not isExist:
#         os.mkdir(output_path)
#     for filename in os.listdir(video_path):
#         if filename != '.DS_Store':
#             video = video_path  + '/' +  filename
#             file_name = os.path.splitext(filename)[0]
#             out_path = output_path
#             out = out_path + '/' +  file_name + '_bin.txt'
#             bi_read_video(video,out) # binary
        
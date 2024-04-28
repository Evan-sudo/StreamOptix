function video2txt (outfile,vid_dec)
    fid=fopen(outfile,'w');%写入文件路径
    [r,c]=size(vid_dec);            % 得到矩阵的行数和列数
     for i=1:r
      for j=1:c
      fprintf(fid,'%1d',vid_dec(i,j));
      end
      fprintf(fid,'\r\n');
     end
    fclose(fid);
end
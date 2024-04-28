import subprocess
import os

folder_path = 'C:\\Users\\liuza\\Desktop\\cross layer\\'



py_files = [file for file in ['bb.py','bola.py','hyb.py','rb.py','mpc.py']]


for py_file in py_files:
    file_path = os.path.join(folder_path, py_file)
    subprocess.run(['python39',file_path], check=True)
import pandas as pd
column_index = 11
def cal_avg(file_path):
    try:
        df = pd.read_csv(file_path, sep = '\t', header = None)
        df = df.drop(df.index[df.iloc[:,column_index]==1])
        print(df)
        avg = df.mean()
        return avg
    except Exception as e:
        print(f"An Error Occurred: {e}")
        return None
    

file_path = './log_sim_hyb'
avgs = cal_avg(file_path)
if avgs is not None:
     print(avgs)
import pandas as pd

def cal_avg(file_path):
    try:
        df = pd.read_csv(file_path, sep = '\t', header = None)
        df = df.drop(df.index[df.iloc[:,11]==1])
        # for c in df.columns:
        #     df[c] = df[c].str.replace(r'[^\d]','',regex=True).astype(float)
        print(df)
        avg = df.mean()
        return avg
    except Exception as e:
        print(f"An Error Occurred: {e}")
        return None
    

file_path = './log_sim_rb'
avgs = cal_avg(file_path)
if avgs is not None:
     print(avgs)
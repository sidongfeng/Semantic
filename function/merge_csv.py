import os
import tqdm
import pandas as pd

FILE = os.listdir("./data/")
try:
    FILE.remove(".DS")
except:
    pass

try:
    df_result = pd.read_csv("./data/"+FILE.pop())
except:
    print("No files in data folder")
    exit()

for f in tqdm.tqdm(FILE):
    df = pd.read_csv("./data/"+f)
    for _, row in df.iterrows():
        id = row["id"]
        # if id != 6204431:
        #     continue
        
        tmp = df_result.loc[df_result["id"]== id]
        # tmp != 0 -> contain id -> update
        # tmp == 0 -> no such id -> append
        if len(tmp) != 0:
            normalize = [x for x in row["normalize"].strip().split("+") if len(x)>0]
            predict = [x for x in row["predict"].strip().split("+") if len(x)>0]
            tmp_predict = [x for x in tmp["predict"].item().strip().split("+") if len(x)>0]
            new_predict = list(set(tmp_predict+predict)-set(normalize))
            if len(new_predict) > 0:
                new_predict_string = "+".join(new_predict)
            else:
                new_predict_string = " "
            df_result.loc[df_result["id"]== id] = new_predict_string
        else:
            df_result = df_result.append(row, ignore_index=True)

fo = open("result.csv","w")
fo.write(df_result.to_csv(index=False))
fo.close()

df = pd.read_csv("result.csv")
print(df)

if __name__ == "__main__":
    None
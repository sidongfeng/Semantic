import os
import tqdm
import pandas as pd

CSV_PATH = "./csv/"
URL_PATH = "./url/"

CSV = os.listdir(CSV_PATH)
try:
    CSV.remove(".DS_Store")
except:
    pass
URL = os.listdir(URL_PATH)
try:
    URL.remove(".DS_Store")
except:
    pass

def merge_csv():
    try:
        csv_result = pd.read_csv(CSV_PATH+CSV.pop())
    except:
        print("No files in data folder")
        exit()
    for f in tqdm.tqdm(CSV):
        df = pd.read_csv(CSV_PATH+f)
        for _, row in df.iterrows():
            id = row["id"]
            # if id != 6204431:
            #     continue
            
            tmp = csv_result.loc[csv_result["id"]== id]
            # tmp != 0 -> contain id -> update
            # tmp == 0 -> no such id -> append
            if len(tmp) != 0:
                normalize = [x for x in row["normalize"].strip().split("+") if len(x)>0]
                predict = [x for x in row["predict"].strip().split("+") if len(x)>0]
                tmp_predict = [x for x in tmp["predict"].item().strip().split("+") if len(x)>0]
                new_predict = [x for x in set(tmp_predict+predict) if x not in normalize]
                if len(new_predict) > 0:
                    new_predict_string = "+".join(new_predict)
                else:
                    new_predict_string = " "
                csv_result.loc[csv_result["id"]== id,"predict"] = new_predict_string
            else:
                csv_result = csv_result.append(row, ignore_index=True)
    return csv_result

def merge_url():
    try:
        url_result = pd.read_csv(URL_PATH+URL.pop(),header=None)
    except:
        print("No files in data folder")
        exit()
    for f in tqdm.tqdm(URL):
        df = pd.read_csv(URL_PATH+f,header=None)
        for _, row in df.iterrows():
            id = row[0]
            # if id != 6204431:
            #     continue
            
            tmp = url_result.loc[url_result[0]== id]
            # tmp != 0 -> contain id -> pass
            # tmp == 0 -> no such id -> append
            if len(tmp) != 0:
                pass
            else:
                url_result = url_result.append(row, ignore_index=True)
    url_result = url_result[[0,2]]
    url_result.columns = ["id","url"]
    return url_result

def merge():
    csv_result = merge_csv()
    # url_result = merge_url()
    # result = pd.merge(csv_result, url_result, on="id")

    fo = open("result.csv","w")
    fo.write(csv_result.to_csv(index=False, encoding='utf-8'))
    fo.close()

if __name__ == "__main__":
    merge()
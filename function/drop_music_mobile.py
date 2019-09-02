import os
import tqdm
import pandas as pd

ui = ["ui", "user interface","user-interface", "user_interface", "user interface design", "user_interface_design", "uidesign", "ui design", "ui_design", "uiuxdesign", "uxuidesign", "uiux design", "uxui design", "uiux_design", "uxui_design", "uiux-design", "uxui-design", "uiux", "uidesigner", "ui_ux", "ui.ux", "ux.ui", "uxui", "ui-ux", "ux-ui", "app-ui", "daily ui", "dailyui", "daily_ui", "30_days_of_ui", "30 days of ui"]

df = pd.read_csv("result.csv")

for i, row in df.iterrows():
    predict = row["predict"]
    predict = predict.split('+')
    if "music" in predict:
        predict.remove("music")
    if "website" in predict:
        predict.remove("website")
    if "mobile" in predict:
        predict.remove("mobile")
    if len(predict) == 0:
        df.at[i,"predict"] = ' '
    else:
        df.at[i,"predict"] = '+'.join(predict)
    
    for x in ui:
        if type(row["origin"]) != str:
            df = df.drop(i)
            break
        if x in row["origin"]:
            break
        if x == ui[-1]:
            df = df.drop(i)
print(df)
fo = open("result1.csv","w")
fo.write(df.to_csv(index=False, encoding='utf-8'))
fo.close()

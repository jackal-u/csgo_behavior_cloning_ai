# 读取当前的视频数据，并为之匹配label
import os, re


files = os.listdir(r"./data")
meta_list = []
for mp4_name in files:
    mp4_path = r"./data" + "/" + mp4_name
    label_name = re.sub(r'\d+_player', '.+_player', mp4_name).strip(".mp4") + '.csv'
    label_folder = "./labels/" + re.sub(r"_round\d+_[t,c]+_tick_\d+_\.\+_player_\d+.csv", "", label_name)
    #print("!!", label_name, label_folder)
    if os.path.isdir(label_folder):
        labels = os.listdir(label_folder)
        label_match = list(filter(lambda v: re.match(label_name, v), labels))
        if len(label_match) == 0:
            continue
        label = label_match[0]
        label_path = label_folder + "/" + label
        print("found ", mp4_path, label_path)
        meta_list.append(mp4_path + " " + label_path)
    else:
        print(os.path.isdir(label_folder), "folder {} not found, skipping it in meta.csv generation".format(label_folder))

with open("./meta.csv", "w") as f:
    f.write("\n".join(meta_list))





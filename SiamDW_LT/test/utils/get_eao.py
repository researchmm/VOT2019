import numpy as np

def main():
    path = '/data/home/v-had/new_unfix.html'
    sta = ['unread', 'ST', 'restore']
    with open(path, "r") as f:
        for line in f:
            if line.startswith("<thead>"):
                line = line.replace('>', " ")
                line = line.replace('<', ' ')
                line = line.split(" ")
                data = []
                for idx, item in enumerate(line):
                    for x in sta:
                        if item.startswith(x):
                            data.append(item)
                    if item.startswith("0"):
                        data.append(item)
                print(len(data))
                res = dict()
                for i in range(len(data)):
                    if i % 2 == 1: continue
                    if res.get(data[i][:-4]) == None:
                        res[data[i][:-4]] = [float(data[i+1])]
                    else:
                        res[data[i][:-4]].append(float(data[i+1]))
                for key in res.keys():
                    data = res[key]
                    data = np.array(data)
                    print(key, data.mean())

if __name__ == '__main__':
    main()
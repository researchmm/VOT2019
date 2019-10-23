import os

def main():
    template_path = "parameter/atom/restore_vgg19_TLVC.py"
    data = []
    with open(template_path, "r") as f:
        for line in f:
            data.append(line)

    for i in range(28, 41):
        file_path = "parameter/atom/restore_vgg19_TLVC_{}.py".format(i)
        with open(file_path, "w") as f:
            for item in data:
                if "new_vgg19_TLVC" in item:
                    print(item)
                    item = item.replace('new_vgg19_TLVC', 'ATOMnet_ep00{:02d}'.format(i))
                f.write(item)

if __name__ == '__main__':
    main()
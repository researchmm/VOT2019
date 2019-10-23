import matplotlib.pyplot as plt
import random

def draw_line(x, y, name):
    for idx, xx in enumerate(x):
        yy = y[idx]
        plt.plot(xx, yy, color=(random.random(), random.random(), random.random()))

    plt.savefig(name)

def main():
    X = []
    Y = []

    x = [0.38,0.359263 ,

0.3559 ,

0.393935,

0.357380 ,

0.348148 ,

0.373181 ,

0.378639 ,

0.392282 ,

0.361640 ,

0.370160 ,

0.385925 ,

0.371413 ,

0.375 ,

0.371292 ]
    y=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    X.append(x)
    Y.append(y)
    draw_line(Y, X, "/data/home/v-had/debug.jpg")

if __name__ == '__main__':
    main()
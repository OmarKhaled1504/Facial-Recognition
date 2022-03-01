import matplotlib.pyplot as plt

with open("/faces/s1/1.pgm", 'rb') as pgmf:
    im = plt.imread(pgmf)

print(im)

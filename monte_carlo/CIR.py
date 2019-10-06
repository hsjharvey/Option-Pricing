import numpy as np

alpha = float(1)
b = float(0.25)
sigma_r = float(0.1)
T = 1
v0 = float(0.1)
v_list = []
n = 100
N = 252

df = 4 * b * alpha / float(sigma_r ** 2)


def CIR():
    for j in range(n):
        v = 0.1
        for i in range(N):
            Lambda = 4 * alpha * np.exp(-alpha / N) * v / ((sigma_r ** 2) * (1 - np.exp(-alpha / N)))
            v = (sigma_r ** 2) * (1 - np.exp(-alpha / N)) / (4 * alpha) * np.random.noncentral_chisquare(df,
                                                                                                         nonc=Lambda)
        v_list.append(v)
    mue = np.mean(v_list)
    std_error = np.std(v_list) / np.sqrt(n)
    print("Mean is: " + str(mue))
    print("Standard error is: " + str(std_error))


if __name__ == '__main__':
    CIR()

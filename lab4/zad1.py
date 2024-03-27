import math
bias1 = 0.80109
bias2 = 0.43529
bias3 = -0.2368


def act_func(x):
    return (1 / (1 + ((math.e) ** (-x))))


def forwardPass(wiek, waga, wzrost):
    wiek_n1 = wiek * -0.46122
    wiek_n2 = wiek * 0.78548

    waga_n1 = waga * 0.97314
    waga_n2 = waga * 2.10584

    wzrost_n1 = wzrost * -0.39203
    wzrost_n2 = wzrost * -0.57847

    h_n1 = wiek_n1 + waga_n1 + wzrost_n1 + bias1
    h_n1_active = act_func(h_n1)
    h_n2 = wiek_n2 + waga_n2 + wzrost_n2 + bias2
    h_n2_active = act_func(h_n2)

    h_n3 = ((h_n1_active * -0.81546) + (h_n2_active * 1.03775)) + bias3

    print(h_n3)


forwardPass(23, 75, 176)

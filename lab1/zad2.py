import numpy as np
import matplotlib.pyplot as plt

def main():
    cel = np.random.randint(50, 341)
    print("Cel znajduje się w odległości:", cel, "metrów")

    v0 = 50 
    h = 100 

    while True:
        alpha = float(input("Podaj kąt strzału w stopniach (0-90): "))
        alpha_rad = np.deg2rad(alpha)

        d = calculate_distance(v0, alpha_rad, h)

        if abs(d - cel) <= 5:
            print("Cel trafiony!")
            break
        else:
            print("Pocisk minął cel o", abs(d - cel), "metrów. Spróbuj ponownie!")

    plot_trajectory(v0, alpha_rad, h, cel)

def calculate_distance(v0, alpha, h):
    distance = (v0 ** 2) * np.sin(2 * alpha) / 9.81
    return distance

def plot_trajectory(v0, alpha, h, cel):
    t_flight = 2 * v0 * np.sin(alpha) / 9.81
    t = np.linspace(0, t_flight, num=1000)

    x = v0 * np.cos(alpha) * t
    y = -0.5 * 9.81 * t**2 + v0 * np.sin(alpha) * t + h

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="Trajektoria pocisku")
    plt.scatter(cel, 0, color='red', label="Cel")
    plt.xlabel("Dystans (m)")
    plt.ylabel("Wysokość (m)")
    plt.title("Trajektoria pocisku Warwolf")
    plt.grid(True)
    plt.legend()
    plt.savefig("trajektoria.png")
    plt.show()

if __name__ == "__main__":
    main()

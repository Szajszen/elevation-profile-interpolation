import numpy as np
import matplotlib.pyplot as plt

def spline_interpolation(x, y, num_points, uniform=True):
    n = len(x)
    # Wybór sposobu generowania punktów
    if uniform:
        x_interp = np.linspace(x[0], x[-1], num_points)  # Punkty interpolacji
    else:
        np.random.seed(0)  # Aby uzyskać powtarzalne wyniki
        x_interp = np.sort(np.concatenate([np.random.uniform(x[0], (x[0] + x[-1]) / 2, num_points // 2),
                                           np.random.uniform((x[0] + x[-1]) / 2, x[-1], num_points - num_points // 2)]))

    # Obliczanie współczynników
    h = np.diff(x)
    A = np.zeros((n, n))
    b = np.zeros(n)

    # Wypełnianie macierzy A i wektora b
    A[0, 0] = 1
    A[n-1, n-1] = 1

    for i in range(1, n-1):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        b[i] = 3 * (y[i+1] - y[i]) / h[i] - 3 * (y[i] - y[i-1]) / h[i-1]

    c = np.linalg.solve(A, b)  # Rozwiązanie układu równań

    # Obliczanie wartości interpolowanej funkcji dla punktów interpolacji
    y_interp = np.zeros(num_points)
    for i in range(n-1):
        mask = np.logical_and(x_interp >= x[i], x_interp <= x[i+1])
        dx = x_interp[mask] - x[i]
        y_interp[mask] = y[i] + (y[i+1] - y[i]) / h[i] * dx + (1/3) * c[i] * dx**2 + (1/6) * (c[i+1] - c[i]) * dx**3

    return x_interp, y_interp

# Odczyt danych z pliku tekstowego
data = np.genfromtxt('grecja-trasa-pod-gore.txt', delimiter=',')
x_data = data[:, 0]
y_data = data[:, 1]

# Wygenerowanie wykresów z równomiernie rozmieszczonymi punktami
num_points_uniform = [ 25, 40,80,120]

for num in num_points_uniform:
    x_interp, y_interp = spline_interpolation(x_data, y_data, num)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, color='grey', label='Dane rzeczywiste')
    plt.scatter(x_interp, y_interp, color='red', label='Węzły interpolacji')
    plt.plot(x_interp, y_interp, label=f'Interpolacja ({num} punktów)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'Interpolacja funkcją sklejaną ({num} punktów)')
    plt.grid(True)
    #plt.savefig(f'sklejanie_rownomiernie_slowacja_{num}.png')
    #plt.show()


noise_levels = [0, 0.5, 1.5, 3.0]  # Poziomy szumu do sprawdzenia

for noise in noise_levels:
    y_data_noisy = y_data + np.random.normal(0, noise, size=y_data.shape)  # Dodanie szumu

    x_interp, y_interp = spline_interpolation(x_data, y_data_noisy, 50, uniform=False)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, color='grey', label='Dane rzeczywiste')
    plt.scatter(x_data, y_data_noisy, color='blue', label='Dane zaszumione')
    plt.scatter(x_interp, y_interp, color='red', label='Węzły interpolacji')
    plt.plot(x_interp, y_interp, label=f'Interpolacja (50 punktów, szum: {noise})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'Interpolacja funkcją sklejaną z  rozmieszczonymi punktami i szumem {noise}')
    plt.grid(True)
    plt.savefig(f'sklejanie_szumy_grecja_{noise}.png')
    plt.show()


# Wygenerowanie wykresów z nierównomiernie rozmieszczonymi punktami
num_points_nonuniform = [25, 40,80,120]

for num in num_points_nonuniform:
    x_interp, y_interp = spline_interpolation(x_data, y_data, num, uniform=False)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, color='grey', label='Dane rzeczywiste')
    plt.scatter(x_interp, y_interp, color='red', label='Węzły interpolacji')
    plt.plot(x_interp, y_interp, label=f'Interpolacja ({num} punktów)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'Interpolacja funkcją sklejaną z nierównomiernie rozmieszczonymi punktami ({num} punktów)')
    plt.grid(True)
    #plt.savefig(f'sklejanie_nierownomiernie_slowacja_{num}.png')
   # plt.show()






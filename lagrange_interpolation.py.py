import numpy as np
import matplotlib.pyplot as plt

# funkcja do obliczania wielomianu interpolacyjnego Lagrange'a
def lagrange_interpolation(x, y, x_val):
    n = len(x)
    y_val = 0
    for i in range(n):
        p = 1
        for j in range(n):
            if i != j:
                p = p * (x_val - x[j]) / (x[i] - x[j])
        y_val += p * y[i]
    return y_val

# wczytanie danych z pliku txt
data = np.loadtxt("slowacja-z-gorki.txt", delimiter=",")
x_data = data[:, 0]
y_data = data[:, 1]

# różne liczby punktów węzłowych do przetestowania
num_points = [4, 12, 15, 20]

# rysowanie wykresów dla różnych liczby punktów węzłowych
for n in num_points:
    # wybór n punktów równomiernie rozłożonych wzdłuż trasy
    indices = np.round(np.linspace(0, len(x_data) - 1, n)).astype(int)
    x_selected = x_data[indices]
    y_selected = y_data[indices]

    # zdefiniowanie punktów do interpolacji
    x_vals = np.linspace(min(x_data), max(x_data), 1000)

    # obliczenie wartości y dla punktów interpolowanych za pomocą wielomianu Lagrange'a
    y_vals = [lagrange_interpolation(x_selected, y_selected, x) for x in x_vals]

    # rysowanie wykresu
    plt.figure()
    plt.scatter(x_data, y_data, color='grey', label='Dane rzeczywiste')  # dane rzeczywiste
    plt.scatter(x_selected, y_selected, color='green', label='Węzły interpolacji')  # węzły interpolacji
    plt.plot(x_vals, y_vals, label='Interpolacja Lagrange')  # wykres interpolowany
    plt.title(f'Interpolacja Lagrangea - {n} punktów')
    plt.xlabel('Odległość')
    plt.ylabel('Wysokość')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'lagrange_rownomiernie_slowacja_{n}.png')
    plt.show()

# rysowanie wykresów dla nierównomiernych punktów węzłowych
for n in num_points:
    # wybór n punktów nierównomiernie rozłożonych wzdłuż trasy
    indices = np.sort(np.random.choice(len(x_data), n, replace=False))
    x_selected = x_data[indices]
    y_selected = y_data[indices]

    # obliczenie wartości y dla punktów interpolowanych za pomocą wielomianu Lagrange'a
    y_vals = [lagrange_interpolation(x_selected, y_selected, x) for x in x_vals]

    # rysowanie wykresu
    plt.figure()
    plt.scatter(x_data, y_data, color='grey', label='Dane rzeczywiste')  # dane rzeczywiste
    plt.scatter(x_selected, y_selected, color='red', label='Nierównomiernie rozmieszczone węzły interpolacji')  # węzły interpolacji
    plt.plot(x_vals, y_vals, label='Interpolacja Lagrange')  # wykres interpolowany
    plt.title(f'Interpolacja Lagrangea - {n} punktów')
    plt.xlabel('Odległość')
    plt.ylabel('Wysokość')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'lagrange_nierownomiernie_slowacja_{n}.png')

    plt.show()






# różne poziomy szumu do przetestowania
noise_levels = [0, 0.5, 1.5, 3.0]

# rysowanie wykresów dla różnych poziomów szumu
for noise in noise_levels:
    # wybór n punktów równomiernie rozłożonych wzdłuż trasy
    indices = np.round(np.linspace(0, len(x_data) - 1, 15)).astype(int)  # przykładowa liczba punktów 20
    x_selected = x_data[indices]

    # dodanie szumu do wartości y
    y_data_noisy = y_data + np.random.normal(0, noise, size=y_data.shape)

    y_selected = y_data_noisy[indices]

    # zdefiniowanie punktów do interpolacji
    x_vals = np.linspace(min(x_data), max(x_data), 1000)

    # obliczenie wartości y dla punktów interpolowanych za pomocą wielomianu Lagrange'a
    y_vals = [lagrange_interpolation(x_selected, y_selected, x) for x in x_vals]

    # rysowanie wykresu
    plt.figure()
    plt.scatter(x_data, y_data, color='grey', label='Dane rzeczywiste')  # dane rzeczywiste
    plt.scatter(x_data, y_data_noisy, color='blue', label='Dane zaszumione')  # dane zaszumione
    plt.scatter(x_selected, y_selected, color='red', label='Węzły interpolacji')  # węzły interpolacji
    plt.plot(x_vals, y_vals, label=f'Interpolacja Lagrangea - 15 punktów, szum: {noise}')  # wykres interpolowany
    plt.title(f'Interpolacja Lagrangea z równomiernie rozmieszczonymi punktami i szumem {noise}')
    plt.xlabel('Odległość')
    plt.ylabel('Wysokość')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'lagrange_szumy_slowacja_{noise}.png')

    plt.show()

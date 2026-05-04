import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def circular_velocity_NFW(r, M_200, c):
    """ Restituisce la velocità di rotazione di un alone di DM, assumendo un profilo di densità NFW. 
    r: array di numpy con i valori dei raggi (in kpc)
    M_200: Massa viriale (in Masse Solari)
    c: Parametro di concentrazione """
    
    # Costanti fisiche
    G = 4.3009e-6           # in kpc/M_sun * (km/s)^2
    H_0 = 70.0 / 1000.0     # H_0 in km/s/kpc

    # Densità critica dell'Universo (M_sun / kpc^3)
    rho_c = 3 * (H_0**2) / (8 * np.pi * G)

    # Scrivo r_200 in funzione di M_200
    r_200 = ((3 * M_200) / (4 * np.pi * 200 * rho_c))**(1/3)
    
    # Scrivo il raggio di scala a in funzione di r_200 e della concentarzione c
    a = r_200 / c

    # Calcolo M(r) come integrale della densità NFW (la soluzione è analitica).
    # Per comododità la scrivo in funzione di M_200.

    def f(x):
        # Per comodità definisco questa fuznione per il calcolo di M(r).
        return np.log(1 + x) - (x / (1 + x))

    M_r = M_200 * (f(r / a) / f(c))

    # Calcolo la circular velocity
    v_c = np.sqrt(G * M_r / r)
    return v_c

def generazione_dati(M_200_true=1.5e12, c_true=12.0, r_min=2, r_max=40, n_punti=25, seed=100):
    """ Genera dei dati simulati dell'osservazione della velocità di rotazione di una galassia, assumendo un profilo di densità NFW. 
    I valori true sono quelli stimati per Andromeda.
    In output restituisce tre array: il primo contiene i raggi a cui sono effettuate le osservazioni, il secondo le velocità misurate e
    il terzo l'errore sulle velocità. """

    np.random.seed(seed)

    # Array contenete i raggi a cui vengono effettuate le osservazioni.
    r_obs = np.linspace(r_min, r_max, n_punti)

    # Velocità teoriche utilizzando la fuznione circular_velocity_NFW definita in precedenza.
    v_th = circular_velocity_NFW(r_obs, M_200_true, c_true)

    # Simulazione degli errori strumentali casuali tra 5 e 15 
    err_v = np.random.uniform(5, 15, size=n_punti)

    # Errori statistici
    v_noise = np.random.normal(loc=0.0, scale=err_v, size=n_punti)

    # Velocità misurate    
    v_obs = v_th + v_noise

    return r_obs, v_obs, err_v

# Generazione dati
r_obs, v_obs, err_v = generazione_dati()

# Fit dei parametri M_200 e c usando la funzione curve_fit (minimi quadrati pesati)
punto_iniziale = [1e11, 10.0]
best_par, cov_matrix = curve_fit(circular_velocity_NFW, r_obs, v_obs, p0=punto_iniziale, sigma=err_v, absolute_sigma=True)

# "Spacchetto" i parametri sfruttando la sintassi Python
M_200_best, c_best = best_par

# Prendo l'errore sui parametri come radici quadrate degli elementi diagonali della matrice di covarianza
err_M_200, err_c = np.sqrt(np.diag(cov_matrix))

print("Parametri stimati")
print(f"M_200 best-fit: ( {M_200_best:.2e} ± {err_M_200:.2e} ) M_sun")
print(f"Concentrazione best-fit: ( {c_best:.2f} ± {err_c:.2f} )")

# Grafici

r_model = np.linspace(0.1, 80, 200)
v_model = circular_velocity_NFW(r_model, M_200_best, c_best)

plt.figure(figsize=(10, 6))

# Plot dei dati con errori
plt.errorbar(r_obs, v_obs, yerr=err_v, fmt='ko', capsize=4, elinewidth=1.5, alpha=0.8, label='Dati (simulati)')

# Plot della curva teorica
plt.plot(r_model, v_model, 'r-', linewidth=2.5, label=f'Best-Fit NFW\n$M_{{200}} = {M_200_best:.1e}\ M_\odot$\n$c = {c_best:.1f}$')

plt.axvspan(40, 80, color='gray', alpha=0.15, label='Regione estrapolata')

plt.xlabel('Raggio $r$ (kpc)', fontsize=13)
plt.ylabel('Circular velocity $V_c$ (km/s)', fontsize=13)
plt.title('Curva di rotazione di M31 usando NFW', fontsize=15)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()
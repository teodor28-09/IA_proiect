"""
comparatie.py - Compara variantele gbest, lbest social si lbest geografic.

Fiecare varianta este rulata de mai multe ori (PSO este stochastic, deci o
singura rulare nu spune mare lucru). Se afiseaza media, deviatia standard,
cel mai bun si cel mai slab rezultat, plus un grafic cu convergenta medie.
Rezultatele sunt bune pentru capitolul "Rezultate" din documentatie.

Rulare:
    python comparatie.py
"""

import statistics

import matplotlib.pyplot as plt

import problema_router as problema
from pso import PSO

NR_RULARI = 30
NR_PARTICULE = 20
NR_ITERATII = 60

VARIANTE = ["gbest", "lbest_social", "lbest_geografic"]


def main():
    rezultate = {}      # varianta -> lista cu fitness-ul final al fiecarei rulari
    convergenta = {}    # varianta -> curba medie de convergenta

    for varianta in VARIANTE:
        finale = []
        curbe = []
        for rulare in range(NR_RULARI):
            pso = PSO(problema.fitness, problema.LIMITE_MIN, problema.LIMITE_MAX,
                      nr_particule=NR_PARTICULE, varianta=varianta, seed=rulare)
            pso.ruleaza(NR_ITERATII)
            finale.append(pso.gbest_fitness)
            curbe.append(pso.istoric_fitness)
        rezultate[varianta] = finale
        # media pe fiecare iteratie
        convergenta[varianta] = [statistics.mean(c[i] for c in curbe)
                                 for i in range(NR_ITERATII + 1)]
        print(f"{varianta} gata.")

    # ---- tabel ----
    print()
    print(f"{NR_RULARI} rulari, {NR_PARTICULE} particule, {NR_ITERATII} iteratii")
    print(f"{'Varianta':<18}{'Media':>8}{'Dev.std':>9}{'Maxim':>8}{'Minim':>8}")
    for varianta, finale in rezultate.items():
        print(f"{varianta:<18}{statistics.mean(finale):>8.2f}"
              f"{statistics.stdev(finale):>9.2f}{max(finale):>8.2f}{min(finale):>8.2f}")

    # ---- grafic ----
    plt.figure(figsize=(8, 5))
    for varianta, curba in convergenta.items():
        plt.plot(curba, label=varianta)
    plt.xlabel("Iteratie")
    plt.ylabel("Acoperire medie a celui mai bun router (%)")
    plt.title(f"Convergenta medie pe {NR_RULARI} rulari")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("comparatie_convergenta.png", dpi=150)
    print("\nGraficul a fost salvat in comparatie_convergenta.png")
    plt.show()


if __name__ == "__main__":
    main()

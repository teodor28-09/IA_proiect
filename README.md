# PSO – Amplasarea optimă a unui router Wi-Fi

Proiect IA – Tema 6: Optimizarea de tip roi de particule (Particle Swarm Optimization).

Căutăm poziția (x, y) a unui router într-un apartament de 20 × 12 m cu pereți
din beton, cărămidă și rigips, astfel încât **cât mai mare parte din suprafață
să aibă semnal ≥ −58 dBm**.

## Structura fișierelor

| Fișier | Ce conține |
|---|---|
| `pso.py` | **Doar algoritmul PSO** (gbest, lbest social, lbest geografic). Nu știe nimic despre routere sau grafice. |
| `problema_router.py` | **Doar problema**: planul apartamentului, modelul de propagare a semnalului, funcția de fitness. |
| `interfata.py` | **Doar UI-ul**: animația 2D a particulelor, graficul de convergență, harta semnalului. |
| `comparatie.py` | Rulează fiecare variantă de 30 de ori și compară rezultatele (tabel + grafic). |

Legătura dintre ele este o singură linie:

```python
pso = PSO(functie_fitness=problema.fitness, limite_min=..., limite_max=..., varianta="gbest")
pso.pas()      # o iterație
```

## Instalare și rulare

```bash
pip install -r requirements.txt
python interfata.py      # animația
python comparatie.py     # comparația gbest / lbest
```

## Algoritmul pe scurt (pso.py)

Pentru fiecare particulă și fiecare dimensiune d:

```
v[d] = w*v[d] + c1*r1*(pbest[d] - x[d]) + c2*r2*(lider[d] - x[d])
x[d] = x[d] + v[d]
```

- `w = 0.729` (inerție), `c1 = c2 = 1.49445`, viteza limitată la 20% din domeniu
- **lider** = depinde de variantă:
  - `gbest` – cea mai bună poziție din tot roiul
  - `lbest_social` – cel mai bun pbest dintre vecinii de index (inel: i−1, i, i+1)
  - `lbest_geografic` – cel mai bun pbest dintre cele mai apropiate 3 particule în spațiu

## Modelul de semnal (problema_router.py)

```
semnal = P_TX − (PL0 + 10·n·log10(d)) − Σ atenuarea pereților traversați
```

P_TX = 20 dBm, PL0 = 40 dB, n = 3, beton 12 dB, cărămidă 7 dB, rigips 3 dB.
Fitness = procentul din punctele unei grile de 0.25 m cu semnal ≥ −58 dBm.

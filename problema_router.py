"""
problema_router.py - Problema amplasarii optime a unui router Wi-Fi intr-un apartament.

Fisierul descrie PROBLEMA (planul apartamentului, modelul de propagare a
semnalului si functia de fitness). Nu stie nimic despre PSO sau interfata.

Model:
  - Apartamentul are LATIME x INALTIME metri si contine pereti interiori.
  - Fiecare perete atenueaza semnalul cu un anumit numar de dB
    (beton > caramida > rigips).
  - Puterea semnalului intr-un punct (in dBm) este:

        semnal = P_TX - [ PL0 + 10 * n * log10(d) ] - suma(atenuari pereti traversati)

    unde d = distanta router-punct, iar al doilea termen este modelul
    "log-distance path loss" folosit pentru propagarea in interior.
  - Un punct este ACOPERIT daca semnalul este >= PRAG_SEMNAL.

Fitness (de MAXIMIZAT):
  procentul din suprafata apartamentului care este acoperita.

Fitness-ul nu are o formula analitica simpla: pentru fiecare pozitie a
routerului se verifica, pentru mii de puncte, ce pereti sunt intersectati
de linia router-punct.
"""

import numpy as np

# ----------------------------------------------------------------------
# Dimensiunile apartamentului (metri)
# ----------------------------------------------------------------------
LATIME = 20.0
INALTIME = 12.0

# ----------------------------------------------------------------------
# Parametrii modelului de propagare
# ----------------------------------------------------------------------
P_TX = 20.0           # puterea de emisie a routerului (dBm)
PL0 = 40.0            # pierderea la 1 metru (dB), tipic pentru 2.4 GHz
N_EXPONENT = 3.0      # exponentul de pierdere pentru interior (2 = spatiu liber)
PRAG_SEMNAL = -58.0   # dBm: sub aceasta valoare consideram semnalul prea slab
                      # (prag strict, pentru streaming/video-call de calitate)

# Atenuarea tipurilor de pereti (dB)
BETON = 12.0
CARAMIDA = 7.0
RIGIPS = 3.0

# ----------------------------------------------------------------------
# Peretii interiori: ((x1, y1), (x2, y2), atenuare)
# Golurile dintre segmente sunt usi.
#
#   y=12 +---------------+-------------+-----------+
#        |               |             |           |
#        |   Dormitor    |  Living     | Dormitor 2|
#        |               |             |           |
#   y=7  +---  ----------+             +----  -----+  y=5 (dreapta)
#        |   Hol         |             |           |
#        |               +----- ---+   | Bucatarie |
#        |   Baie        |  Birou  |   |           |
#   y=0  +---------------+---------+---+-----------+
#        x=0            x=7      x=12 x=14        x=20
# ----------------------------------------------------------------------
PERETI = [
    # perete portant din beton, vertical, la x = 7 (usa intre y=5 si y=6.5)
    ((7.0, 0.0), (7.0, 5.0), BETON),
    ((7.0, 6.5), (7.0, 12.0), BETON),

    # perete din caramida, vertical, la x = 14 (usa intre y=3 si y=4.5)
    ((14.0, 0.0), (14.0, 3.0), CARAMIDA),
    ((14.0, 4.5), (14.0, 12.0), CARAMIDA),

    # dormitorul 1 (stanga sus), perete orizontal la y = 7 (usa intre x=3 si x=4.5)
    ((0.0, 7.0), (3.0, 7.0), CARAMIDA),
    ((4.5, 7.0), (7.0, 7.0), CARAMIDA),

    # baia (stanga jos), perete orizontal la y = 3.5 - faianta + tevi, atenuare mare
    ((0.0, 3.5), (5.0, 3.5), BETON),

    # biroul (centru jos), pereti de rigips
    ((7.0, 4.0), (9.0, 4.0), RIGIPS),
    ((10.5, 4.0), (12.0, 4.0), RIGIPS),
    ((12.0, 0.0), (12.0, 4.0), RIGIPS),

    # dormitorul 2 (dreapta sus), perete orizontal la y = 5 (usa intre x=16 si x=17.5)
    ((14.0, 5.0), (16.0, 5.0), CARAMIDA),
    ((17.5, 5.0), (20.0, 5.0), CARAMIDA),
]

# ----------------------------------------------------------------------
# Punctele in care masuram semnalul: o grila cu pasul de 0.25 m
# ----------------------------------------------------------------------
PAS_GRILA = 0.25
_xs = np.arange(PAS_GRILA / 2, LATIME, PAS_GRILA)
_ys = np.arange(PAS_GRILA / 2, INALTIME, PAS_GRILA)
_GX, _GY = np.meshgrid(_xs, _ys)
PUNCTE = np.column_stack([_GX.ravel(), _GY.ravel()])   # matrice N x 2
FORMA_GRILA = _GX.shape                                 # (randuri, coloane)

# Limitele spatiului de cautare pentru PSO: routerul poate fi oriunde in apartament
LIMITE_MIN = [0.0, 0.0]
LIMITE_MAX = [LATIME, INALTIME]


def _produs_vectorial(ax, ay, bx, by):
    return ax * by - ay * bx


def _intersecteaza_perete(router, puncte, a, b):
    """
    Pentru fiecare punct, verifica daca segmentul router -> punct
    intersecteaza peretele a -> b. Intoarce un vector de True/False.

    Doua segmente se intersecteaza daca capetele fiecaruia sunt de o parte
    si de alta a celuilalt (testul de orientare cu produs vectorial).
    """
    rx, ry = router
    px, py = puncte[:, 0], puncte[:, 1]
    ax, ay = a
    bx, by = b

    # de ce parte a peretelui sunt routerul si punctul?
    d1 = _produs_vectorial(bx - ax, by - ay, rx - ax, ry - ay)
    d2 = _produs_vectorial(bx - ax, by - ay, px - ax, py - ay)
    # de ce parte a liniei router-punct sunt capetele peretelui?
    d3 = _produs_vectorial(px - rx, py - ry, ax - rx, ay - ry)
    d4 = _produs_vectorial(px - rx, py - ry, bx - rx, by - ry)

    return (d1 * d2 < 0) & (d3 * d4 < 0)


def calculeaza_semnal(router, puncte=PUNCTE):
    """Intoarce puterea semnalului (dBm) in fiecare punct, pentru un router la pozitia data."""
    router = np.asarray(router, dtype=float)

    # 1. pierderea datorata distantei
    distanta = np.linalg.norm(puncte - router, axis=1)
    distanta = np.maximum(distanta, 0.5)   # evitam log(0) langa router
    pierdere_distanta = PL0 + 10 * N_EXPONENT * np.log10(distanta)

    # 2. pierderea datorata peretilor traversati
    pierdere_pereti = np.zeros(len(puncte))
    for a, b, atenuare in PERETI:
        traverseaza = _intersecteaza_perete(router, puncte, a, b)
        pierdere_pereti += traverseaza * atenuare

    return P_TX - pierdere_distanta - pierdere_pereti


def fitness(pozitie):
    """
    FUNCTIA DE FITNESS pentru PSO.
    pozitie = [x, y] a routerului.
    Intoarce procentul de suprafata acoperita (0..100) - de maximizat.
    """
    semnal = calculeaza_semnal(pozitie)
    acoperit = semnal >= PRAG_SEMNAL
    return 100.0 * np.mean(acoperit)


def harta_semnal(pozitie):
    """Semnalul sub forma de matrice 2D (pentru desenarea unui heatmap)."""
    return calculeaza_semnal(pozitie).reshape(FORMA_GRILA)


def harta_fitness(pas=0.5):
    """
    Calculeaza fitness-ul pentru routere plasate pe o grila (pentru fundalul
    animatiei - arata unde sunt zonele bune si optimele locale).
    Este doar pentru vizualizare; PSO NU foloseste aceasta harta.
    """
    xs = np.arange(pas / 2, LATIME, pas)
    ys = np.arange(pas / 2, INALTIME, pas)
    harta = np.zeros((len(ys), len(xs)))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            harta[i, j] = fitness([x, y])
    return harta

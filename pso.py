"""
pso.py - Algoritmul de optimizare cu roi de particule (Particle Swarm Optimization).

Fisierul contine DOAR logica algoritmului. Nu stie nimic despre routere,
grafice sau interfata. Primeste o functie de fitness si limitele spatiului
de cautare si cauta pozitia cu fitness MAXIM.

Variante implementate:
    "gbest"            - fiecare particula urmeaza cea mai buna pozitie din TOT roiul
    "lbest_social"     - fiecare particula urmeaza cea mai buna pozitie dintre
                         vecinii ei de INDEX (topologie inel: i-1, i, i+1)
    "lbest_geografic"  - fiecare particula urmeaza cea mai buna pozitie dintre
                         cele mai APROPIATE particule in spatiu (distanta euclidiana)

Formulele (varianta standard cu inertie):
    v = w*v + c1*r1*(pbest - x) + c2*r2*(lider - x)
    x = x + v
"""

import math
import random


class Particula:
    """O particula = o solutie candidat care se misca prin spatiul de cautare."""

    def __init__(self, pozitie, viteza):
        self.pozitie = pozitie            # lista de numere reale, ex: [x, y]
        self.viteza = viteza              # lista de numere reale, aceeasi dimensiune
        self.fitness = -math.inf          # fitness-ul pozitiei curente
        self.best_pozitie = list(pozitie) # pbest: cea mai buna pozitie vizitata
        self.best_fitness = -math.inf     # fitness-ul lui pbest


class PSO:
    def __init__(self, functie_fitness, limite_min, limite_max,
                 nr_particule=30, varianta="gbest",
                 w=0.729, c1=1.49445, c2=1.49445,
                 marime_vecinatate=3, seed=None):
        """
        functie_fitness   - functie care primeste o lista [x1, ..., xn] si intoarce un numar
                            (cu cat mai MARE, cu atat mai bine)
        limite_min/max    - liste cu limitele fiecarei dimensiuni
        varianta          - "gbest", "lbest_social" sau "lbest_geografic"
        w                 - inertia (cat de mult isi pastreaza particula directia)
        c1                - coeficientul cognitiv (atractia spre propriul pbest)
        c2                - coeficientul social (atractia spre lider)
        marime_vecinatate - cate particule are o vecinatate lbest (inclusiv particula insasi)
        """
        if seed is not None:
            random.seed(seed)

        self.functie_fitness = functie_fitness
        self.limite_min = limite_min
        self.limite_max = limite_max
        self.dimensiune = len(limite_min)
        self.varianta = varianta
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.marime_vecinatate = marime_vecinatate

        # Viteza maxima pe fiecare dimensiune = 20% din latimea domeniului.
        # Fara limitare, particulele pot "exploda" in afara domeniului.
        self.viteza_max = [0.2 * (limite_max[d] - limite_min[d])
                           for d in range(self.dimensiune)]

        # Cea mai buna solutie gasita vreodata de intregul roi
        self.gbest_pozitie = None
        self.gbest_fitness = -math.inf

        self.iteratie = 0
        self.istoric_fitness = []  # gbest_fitness dupa fiecare iteratie (pentru grafic)

        # --- 1. Initializare aleatoare a particulelor ---
        self.particule = []
        for _ in range(nr_particule):
            pozitie = [random.uniform(limite_min[d], limite_max[d])
                       for d in range(self.dimensiune)]
            viteza = [random.uniform(-self.viteza_max[d], self.viteza_max[d])
                      for d in range(self.dimensiune)]
            self.particule.append(Particula(pozitie, viteza))

        # --- 2. Prima evaluare ---
        self._evalueaza_toate()
        self.istoric_fitness.append(self.gbest_fitness)

    # ------------------------------------------------------------------
    # Evaluare
    # ------------------------------------------------------------------
    def _evalueaza_toate(self):
        """Calculeaza fitness-ul fiecarei particule si actualizeaza pbest si gbest."""
        for p in self.particule:
            p.fitness = self.functie_fitness(p.pozitie)

            # actualizare pbest
            if p.fitness > p.best_fitness:
                p.best_fitness = p.fitness
                p.best_pozitie = list(p.pozitie)

            # actualizare gbest (pastrat pentru raportare la toate variantele)
            if p.fitness > self.gbest_fitness:
                self.gbest_fitness = p.fitness
                self.gbest_pozitie = list(p.pozitie)

    # ------------------------------------------------------------------
    # Vecinatati: cine este "liderul" pe care il urmeaza particula i?
    # ------------------------------------------------------------------
    def _vecini_social(self, i):
        """Topologie inel: vecinii sunt particulele cu indici apropiati.
        Pentru marime 3: i-1, i, i+1 (circular). Nu depinde de pozitie."""
        n = len(self.particule)
        raza = self.marime_vecinatate // 2
        return [(i + k) % n for k in range(-raza, raza + 1)]

    def _vecini_geografic(self, i):
        """Vecinii sunt cele mai apropiate particule in spatiul de cautare
        (inclusiv particula i, aflata la distanta 0). Se recalculeaza la fiecare iteratie."""
        xi = self.particule[i].pozitie
        distante = []
        for j, p in enumerate(self.particule):
            d = math.dist(xi, p.pozitie)
            distante.append((d, j))
        distante.sort()
        return [j for _, j in distante[:self.marime_vecinatate]]

    def _lider(self, i):
        """Intoarce pozitia pe care trebuie sa o urmeze particula i."""
        if self.varianta == "gbest":
            return self.gbest_pozitie

        if self.varianta == "lbest_social":
            vecini = self._vecini_social(i)
        elif self.varianta == "lbest_geografic":
            vecini = self._vecini_geografic(i)
        else:
            raise ValueError("Varianta necunoscuta: " + self.varianta)

        # cel mai bun pbest dintre vecini
        cel_mai_bun = max(vecini, key=lambda j: self.particule[j].best_fitness)
        return self.particule[cel_mai_bun].best_pozitie

    # ------------------------------------------------------------------
    # O iteratie a algoritmului
    # ------------------------------------------------------------------
    def pas(self):
        """Executa o iteratie: actualizeaza vitezele si pozitiile, apoi evalueaza."""

        # Liderii se calculeaza INAINTE de a muta particulele,
        # ca toate particulele sa foloseasca aceeasi "stare" a roiului.
        lideri = [self._lider(i) for i in range(len(self.particule))]

        for i, p in enumerate(self.particule):
            lider = lideri[i]
            for d in range(self.dimensiune):
                r1 = random.random()
                r2 = random.random()

                inertie = self.w * p.viteza[d]
                cognitiv = self.c1 * r1 * (p.best_pozitie[d] - p.pozitie[d])
                social = self.c2 * r2 * (lider[d] - p.pozitie[d])
                v = inertie + cognitiv + social

                # limitare viteza
                v = max(-self.viteza_max[d], min(self.viteza_max[d], v))

                # noua pozitie
                x = p.pozitie[d] + v

                # daca a iesit din domeniu, o lipim de margine si ii oprim viteza
                if x < self.limite_min[d]:
                    x = self.limite_min[d]
                    v = 0.0
                elif x > self.limite_max[d]:
                    x = self.limite_max[d]
                    v = 0.0

                p.viteza[d] = v
                p.pozitie[d] = x

        self._evalueaza_toate()
        self.iteratie += 1
        self.istoric_fitness.append(self.gbest_fitness)

    def ruleaza(self, nr_iteratii):
        """Ruleaza algoritmul complet, fara interfata. Intoarce (pozitie, fitness)."""
        for _ in range(nr_iteratii):
            self.pas()
        return self.gbest_pozitie, self.gbest_fitness

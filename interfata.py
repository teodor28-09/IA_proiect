"""
interfata.py - Interfata grafica: animatia 2D a particulelor pe planul apartamentului.

Aici este DOAR partea de afisare. Algoritmul vine din pso.py,
iar problema (pereti, semnal, fitness) din problema_router.py.

Rulare:
    python interfata.py

In fereastra:
  - alegi varianta (gbest / lbest social / lbest geografic)
  - setezi numarul de particule si de iteratii
  - apesi "Start"
  - stanga: particulele se misca pe planul apartamentului
            (fundalul arata cat de buna ar fi fiecare pozitie a routerului)
  - dreapta sus: evolutia celui mai bun fitness
  - dreapta jos: harta semnalului pentru cea mai buna pozitie gasita
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, TextBox

import problema_router as problema
from pso import PSO

VARIANTE = {
    "gbest": "gbest",
    "lbest social (inel)": "lbest_social",
    "lbest geografic": "lbest_geografic",
}


def deseneaza_pereti(ax):
    """Deseneaza conturul apartamentului si peretii interiori."""
    culori = {problema.BETON: "black", problema.CARAMIDA: "saddlebrown",
              problema.RIGIPS: "gray"}
    ax.plot([0, problema.LATIME, problema.LATIME, 0, 0],
            [0, 0, problema.INALTIME, problema.INALTIME, 0],
            color="black", linewidth=3)
    for (x1, y1), (x2, y2), atenuare in problema.PERETI:
        ax.plot([x1, x2], [y1, y2], color=culori.get(atenuare, "black"),
                linewidth=3 if atenuare >= problema.BETON else 2)


class Aplicatie:
    def __init__(self):
        print("Calculez harta de fundal (cateva secunde)...")
        self.harta_fitness = problema.harta_fitness(pas=0.5)
        self.ruleaza = False

        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title("PSO - amplasarea optima a routerului Wi-Fi")

        # zonele figurii
        self.ax_plan = self.fig.add_axes([0.04, 0.30, 0.55, 0.62])
        self.ax_conv = self.fig.add_axes([0.66, 0.58, 0.31, 0.34])
        self.ax_semnal = self.fig.add_axes([0.66, 0.12, 0.31, 0.36])

        # --- controale ---
        ax_radio = self.fig.add_axes([0.04, 0.04, 0.17, 0.17])
        ax_radio.set_title("Varianta PSO", fontsize=9)
        self.radio = RadioButtons(ax_radio, list(VARIANTE.keys()))

        ax_part = self.fig.add_axes([0.33, 0.15, 0.06, 0.05])
        self.txt_particule = TextBox(ax_part, "Particule ", initial="20")
        ax_iter = self.fig.add_axes([0.33, 0.07, 0.06, 0.05])
        self.txt_iteratii = TextBox(ax_iter, "Iteratii ", initial="60")

        ax_start = self.fig.add_axes([0.44, 0.11, 0.07, 0.06])
        self.btn_start = Button(ax_start, "Start")
        self.btn_start.on_clicked(self.porneste)

        ax_stop = self.fig.add_axes([0.52, 0.11, 0.07, 0.06])
        self.btn_stop = Button(ax_stop, "Stop")
        self.btn_stop.on_clicked(self.opreste)

        self.deseneaza_plan_gol()
        plt.show()

    # ------------------------------------------------------------------
    def deseneaza_plan_gol(self):
        ax = self.ax_plan
        ax.clear()
        ax.imshow(self.harta_fitness, origin="lower", cmap="Greens", alpha=0.6,
                  extent=[0, problema.LATIME, 0, problema.INALTIME])
        deseneaza_pereti(ax)
        ax.set_xlim(-0.3, problema.LATIME + 0.3)
        ax.set_ylim(-0.3, problema.INALTIME + 0.3)
        ax.set_aspect("equal")
        ax.set_title("Planul apartamentului (verde inchis = pozitie buna pentru router)")

        self.ax_conv.clear()
        self.ax_conv.set_title("Cel mai bun fitness")
        self.ax_conv.set_xlabel("Iteratie")
        self.ax_conv.set_ylabel("Acoperire (%)")
        self.ax_conv.grid(True)

        self.ax_semnal.clear()
        self.ax_semnal.set_title("Semnal pentru cea mai buna pozitie")
        self.fig.canvas.draw_idle()

    def opreste(self, _event):
        self.ruleaza = False

    def porneste(self, _event):
        if self.ruleaza:
            return
        try:
            nr_particule = int(self.txt_particule.text)
            nr_iteratii = int(self.txt_iteratii.text)
        except ValueError:
            print("Numarul de particule si de iteratii trebuie sa fie intregi.")
            return

        varianta = VARIANTE[self.radio.value_selected]

        # ---- aici se creeaza algoritmul; interfata doar il apeleaza ----
        pso = PSO(functie_fitness=problema.fitness,
                  limite_min=problema.LIMITE_MIN,
                  limite_max=problema.LIMITE_MAX,
                  nr_particule=nr_particule,
                  varianta=varianta)

        self.deseneaza_plan_gol()
        ax = self.ax_plan
        puncte = ax.scatter([], [], s=40, c="blue", edgecolors="white", zorder=5, label="particule")
        sageti = None
        cel_mai_bun = ax.scatter([], [], s=250, c="red", marker="*", edgecolors="black", zorder=6, label="gbest")
        ax.legend(loc="upper right", fontsize=8)
        linie_conv, = self.ax_conv.plot([], [], color="tab:blue")
        self.ax_conv.set_xlim(0, nr_iteratii)
        self.ax_conv.set_ylim(0, 100)

        self.ruleaza = True
        for _ in range(nr_iteratii + 1):
            if not self.ruleaza or not plt.fignum_exists(self.fig.number):
                break

            # --- desenam starea curenta a roiului ---
            xs = [p.pozitie[0] for p in pso.particule]
            ys = [p.pozitie[1] for p in pso.particule]
            vx = [p.viteza[0] for p in pso.particule]
            vy = [p.viteza[1] for p in pso.particule]
            puncte.set_offsets(list(zip(xs, ys)))
            if sageti is not None:
                sageti.remove()
            sageti = ax.quiver(xs, ys, vx, vy, angles="xy", scale_units="xy", scale=1, color="blue", alpha=0.4, width=0.003)
            cel_mai_bun.set_offsets([pso.gbest_pozitie])

            linie_conv.set_data(range(len(pso.istoric_fitness)), pso.istoric_fitness)
            ax.set_title(f"{self.radio.value_selected} | iteratia {pso.iteratie} | "
                         f"cea mai buna acoperire: {pso.gbest_fitness:.1f}% la "
                         f"({pso.gbest_pozitie[0]:.2f}, {pso.gbest_pozitie[1]:.2f}) m")
            plt.pause(0.08)

            if pso.iteratie < nr_iteratii:
                pso.pas()   # <-- O iteratie a algoritmului PSO

        self.ruleaza = False
        if plt.fignum_exists(self.fig.number):
            self.deseneaza_semnal(pso.gbest_pozitie)
            print(f"[{varianta}] Rezultat: router la ({pso.gbest_pozitie[0]:.2f}, "
                  f"{pso.gbest_pozitie[1]:.2f}) m, acoperire {pso.gbest_fitness:.2f}%")

    def deseneaza_semnal(self, pozitie):
        ax = self.ax_semnal
        ax.clear()
        harta = problema.harta_semnal(pozitie)
        img = ax.imshow(harta, origin="lower", cmap="RdYlGn", vmin=-85, vmax=-35,
                        extent=[0, problema.LATIME, 0, problema.INALTIME])
        ax.contour(harta, levels=[problema.PRAG_SEMNAL], colors="black",
                   linewidths=1, linestyles="--",
                   extent=[0, problema.LATIME, 0, problema.INALTIME])
        deseneaza_pereti(ax)
        ax.plot(pozitie[0], pozitie[1], "b*", markersize=15)
        ax.set_aspect("equal")
        ax.set_title(f"Semnal (dBm); linia punctata = prag {problema.PRAG_SEMNAL:.0f} dBm",
                     fontsize=9)
        if not hasattr(self, "colorbar"):
            self.colorbar = self.fig.colorbar(img, ax=ax, fraction=0.03)
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    Aplicatie()

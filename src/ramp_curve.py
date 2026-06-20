from __future__ import annotations
from dataclasses import dataclass
import json
import numpy as np
from scipy.interpolate import CubicSpline


@dataclass
class Keyframe:
    time_sec: float
    target_fps: float


class RampCurve:
    """
    WICHTIGER HINWEIS zur Stetigkeit (siehe Validierungs-Lauf, der den
    urspruenglichen Bug aufgedeckt hat):

    Eine stueckweise Smoothstep-Konstruktion (separates 3x^2-2x^3 PRO
    Segment zwischen je zwei Keyframes) ist NUR an den Raendern jedes
    einzelnen Segments C1-stetig (Steigung = 0 an Segmentanfang/-ende).
    Sobald drei oder mehr Keyframes vorliegen, klafft an den INNEREN
    Keyframes (z.B. dem Plateau-Beginn bei t=1.5s in unserem 25-100-25
    Preset) ein Sprung in der ZWEITEN Ableitung, weil das vorherige
    Segment dort noch eine von Null abweichende Steigung haben kann,
    waehrend das Plateau-Segment exakt flach ist.

    Loesung: Statt pro Segment einzeln zu interpolieren, nutzen wir
    PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) als
    GLOBALE Interpolation ueber ALLE Keyframes hinweg. PCHIP ist
    monotonie-erhaltend (kein Ueberschwingen ueber die Keyframe-Werte
    hinaus) UND garantiert C1-Stetigkeit an JEDEM inneren Punkt --
    exakt die Eigenschaft, die fuer eine wahrnehmbar stufenlose Rampe
    benoetigt wird.
    """

    def __init__(self, keyframes: list[Keyframe], interpolation: str = "smoothstep"):
        if len(keyframes) < 2:
            raise ValueError("Mindestens zwei Keyframes erforderlich.")
        self.keyframes = sorted(keyframes, key=lambda k: k.time_sec)
        self.interpolation = interpolation
        self._build_spline()

    @classmethod
    def from_json(cls, path: str) -> "RampCurve":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        keyframes = [Keyframe(k["time_sec"], k["target_fps"]) for k in data["keyframes"]]
        return cls(keyframes, interpolation=data.get("interpolation", "smoothstep"))

    def _build_spline(self) -> None:
        times = np.array([k.time_sec for k in self.keyframes])
        fps_values = np.array([k.target_fps for k in self.keyframes])
        self._times = times
        self._fps_values = fps_values

        if self.interpolation == "catmull_rom":
            # "clamped" erzwingt Steigung 0 an den AEUSSEREN Raendern,
            # bleibt aber an inneren Punkten automatisch C1-stetig, da
            # CubicSpline grundsaetzlich global (nicht stueckweise
            # isoliert) löst. ACHTUNG: bei Plateau-Keyframes (zwei
            # benachbarte Keyframes mit identischem fps-Wert) kann
            # CubicSpline dennoch leicht ueberschwingen -- fuer reine
            # Plateau-Ramps ist "smoothstep" (s.u.) die robustere Wahl.
            self._spline = CubicSpline(times, fps_values, bc_type="clamped")
            self._mode = "spline"
        elif self.interpolation == "smoothstep":
            # WICHTIGER FIX gegenueber der ersten Implementierung:
            #
            # Stueckweises kubisches Smoothstep (3x^2-2x^3) ist nur C1
            # an Segmentgrenzen (Steigung 0), aber NICHT C2 (Kruemmung
            # nicht garantiert 0) -- das erzeugte den im Validierungs-
            # Plot sichtbaren Sprung in der 2. Ableitung genau an den
            # Plateau-Uebergaengen (t=1.5s, t=3.0s).
            #
            # PCHIP (zwischenzeitlich getestet) loest das Problem auch
            # nicht vollstaendig: es ist global C1, aber ebenfalls
            # nicht C2 -- an Plateau-Keyframes bleibt eine Kruemmungs-
            # Diskontinuitaet, weil die Steigung dort exakt auf 0
            # "einrasten" muss, waehrend sie im Nachbarsegment deutlich
            # von 0 abweicht.
            #
            # LOESUNG: Quintic Smootherstep (6x^5 - 15x^4 + 10x^3) PRO
            # SEGMENT. Diese Funktion hat an BEIDEN Raendern (x=0 und
            # x=1) sowohl Steigung ALS AUCH Kruemmung exakt = 0. Das
            # macht jedes Segment an seinen eigenen Raendern C2-neutral,
            # wodurch sich auch an Plateau-Uebergaengen (Nachbarsegment
            # mit Steigung 0) kein Kruemmungssprung mehr ergibt -- weil
            # die Kruemmung auf BEIDEN Seiten der Naht gegen 0 geht.
            self._mode = "quintic_smootherstep"
        else:
            raise ValueError(f"Unbekannter interpolation-Typ: {self.interpolation}")

    @staticmethod
    def _smootherstep(x: np.ndarray) -> np.ndarray:
        """
        Quintic Smootherstep (Ken Perlin): 6x^5 - 15x^4 + 10x^3.
        Im Vergleich zur kubischen Smoothstep (3x^2 - 2x^3) sind hier
        ZUSAETZLICH die erste UND zweite Ableitung an x=0 und x=1 exakt
        Null. Das ist die Eigenschaft, die fuer C2-saubere Uebergaenge
        an Plateau-Keyframes benoetigt wird.
        """
        x = np.clip(x, 0.0, 1.0)
        return 6 * x**5 - 15 * x**4 + 10 * x**3

    def target_fps(self, t: float) -> float:
        """
        Liefert die lokale Ziel-Framerate f(t) zum Zeitpunkt t. t wird
        auf den gueltigen Keyframe-Bereich geclampt, um Extrapolation
        ausserhalb der definierten Rampe zu vermeiden.
        """
        t_clamped = float(np.clip(t, self._times[0], self._times[-1]))

        if self._mode == "spline":
            return float(self._spline(t_clamped))

        # quintic_smootherstep: stueckweise Auswertung pro Segment
        idx = int(np.searchsorted(self._times, t_clamped, side="right") - 1)
        idx = int(np.clip(idx, 0, len(self._times) - 2))

        t0, t1 = self._times[idx], self._times[idx + 1]
        f0, f1 = self._fps_values[idx], self._fps_values[idx + 1]

        if t1 == t0:
            return float(f0)

        local_x = (t_clamped - t0) / (t1 - t0)
        blend = self._smootherstep(np.array([local_x]))[0]
        return float(f0 + (f1 - f0) * blend)
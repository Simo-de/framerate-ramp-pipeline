"""
timing_table.py
----------------
Zwei Modi:

mode="speedramp":
    Klassisches Retiming. output_fps kann von source_fps abweichen.
    Die fps-Kurve steuert Zeitdehnung. 25fps-Kurve = 4x Zeitlupe.

mode="motion_grade":
    Echtes Motion Grading bei konstanter 1x-Geschwindigkeit.
    output_fps wird intern IMMER auf source_fps gezwungen.
    Pro Input-Frame wird GENAU EIN Output-Frame erzeugt (1:1-Mapping).
    Die fps-Kurve steuert ausschliesslich:
      - blur_window_frames: wie viele Input-Frames zu einem Blur-
        Composite zusammengefaltet werden (bei 25fps-Kurve: 4 Frames)
      - hold_count: wie oft das Composite im 100fps-Container wiederholt
        wird (bei 25fps-Kurve: 4x, also 1 Composite fuer je 4 Frames)
    Ergebnis: Container laeuft mit source_fps (100fps), Clip ist exakt
    gleich lang wie das Original, aber der visuelle Look wechselt
    zwischen "knackig 100fps" und "geblurrt 25fps-Charakter".
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from src.ramp_curve import RampCurve


class TimingTableBuilder:
    def __init__(
        self,
        ramp: RampCurve,
        source_fps: float,
        output_fps: float = 25.0,
        mode: str = "motion_grade",
    ):
        if mode not in ("speedramp", "motion_grade"):
            raise ValueError(f"Unbekannter Modus: '{mode}'. Erlaubt: 'speedramp', 'motion_grade'.")
        self.ramp = ramp
        self.source_fps = source_fps
        self.mode = mode
        # Im motion_grade-Modus ist output_fps immer == source_fps.
        # Das ist der mathematische Kern des Fixes: 1 Input-Frame = 1 Output-Frame.
        self.output_fps = source_fps if mode == "motion_grade" else output_fps

    def build(self, total_duration_sec: float) -> pd.DataFrame:
        if self.mode == "speedramp":
            return self._build_speedramp(total_duration_sec)
        else:
            return self._build_motion_grade(total_duration_sec)

    def _build_speedramp(self, total_duration_sec: float) -> pd.DataFrame:
        """Unveraendertes bisheriges Verhalten."""
        n_output_frames = int(round(total_duration_sec * self.output_fps))
        output_times = np.arange(n_output_frames) / self.output_fps
        local_fps = np.array([self.ramp.target_fps(t) for t in output_times])
        velocity_factor = local_fps / self.source_fps

        dt_output = 1.0 / self.output_fps
        source_time = np.cumsum(velocity_factor * dt_output)
        source_time = source_time - source_time[0] + (velocity_factor[0] * dt_output)
        source_time = np.concatenate([[0.0], source_time[:-1]])

        source_frame_float = source_time * self.source_fps
        source_frame_floor = np.floor(source_frame_float).astype(int)
        source_frame_ceil  = np.clip(source_frame_floor + 1, 0, int(total_duration_sec * self.source_fps) - 1)
        rife_timestep = source_frame_float - source_frame_floor

        return pd.DataFrame({
            "output_frame_idx":   np.arange(n_output_frames),
            "output_time_sec":    output_times,
            "source_time_sec":    source_time,
            "source_frame_floor": source_frame_floor,
            "source_frame_ceil":  source_frame_ceil,
            "rife_timestep":      rife_timestep,
            "local_target_fps":   local_fps,
            "hold_count":         np.ones(n_output_frames, dtype=int),
            "blur_window_frames": np.ones(n_output_frames, dtype=float),
            "is_hold_frame":      np.zeros(n_output_frames, dtype=bool),
            "mode":               "speedramp",
        })

    def _build_motion_grade(self, total_duration_sec: float) -> pd.DataFrame:
        """
        Kernlogik des echten Motion Gradings.

        Fuer jeden Input-Frame i bei source_fps (100fps):
          - local_target_fps(t) bestimmt den gewuenschten Look
          - blur_window = source_fps / local_target_fps Quell-Frames
            werden zu einem Composite gefaltet (bei 25fps: 4 Frames)
          - hold_count = round(source_fps / local_target_fps) bestimmt,
            wie oft dieses Composite im Output wiederholt wird

        Wichtig: hold_count ist eine GANZZAHL (muss es sein, weil
        Frames nicht teilbar sind). Die Nicht-Ganzzahligkeit der Rampe
        (z.B. 33.3fps wuerde 3.0003 Hold-Frames benoetigen) wird durch
        Rundung gehandhabt -- dasselbe, was echter Pulldown seit
        Jahrzehnten macht. Visuell nicht wahrnehmbar bei glatten Rampen.

        Die Ausgabe-Tabelle hat eine Zeile pro OUTPUT-Frame (also
        source_fps * duration Zeilen = gleich viele wie Input-Frames).
        Jede Zeile traegt:
          - source_center_frame: der Input-Frame in der Mitte des
            Blur-Fensters (der "repraesentative" Frame)
          - blur_window_frames:  Breite des Blur-Fensters in Frames
          - hold_count:          wie oft dieser Composite ausgegeben wird
          - is_hold_frame:       True fuer die Wiederholungen (nicht den
            ersten Frame des Hold-Blocks) -- hilfreich um im Render-Pass
            nur einmal zu rendern und N-mal zu kopieren
        """
        n_source_frames = int(round(total_duration_sec * self.source_fps))
        source_frame_indices = np.arange(n_source_frames)
        source_times = source_frame_indices / self.source_fps

        # Lokale Ziel-FPS pro Quell-Frame-Zeitpunkt
        local_fps = np.array([self.ramp.target_fps(t) for t in source_times])
        # Clamp: lokale fps nie hoeher als source_fps (waere sinnlos)
        # und nie unter 1fps (vermeidet Division-durch-Null und
        # astronomische Hold-Counts)
        local_fps = np.clip(local_fps, 1.0, self.source_fps)

        # Blur-Fenstergröße in Quell-Frames (kann nicht-ganzzahlig sein)
        blur_window = self.source_fps / local_fps  # z.B. 100/25 = 4.0

        # Hold-Count: wie viele Output-Frames zeigen diesen Blur-Composite?
        # Wir runden auf die naechste ganze Zahl. Das ist der Pulldown-
        # Kompromiss (keine halben Frames).
        hold_count = np.clip(np.round(blur_window).astype(int), 1, int(self.source_fps))

        # Jetzt bauen wir die Output-Frame-Tabelle auf. Wir iterieren
        # durch Bloecke: jeder "Anker-Frame" definiert einen Blur-
        # Composite, der hold_count[i]-mal in den Output geschrieben wird.
        # Wichtig: Wir springen im Input um hold_count vorwaerts, damit
        # kein Input-Frame doppelt als Anker genutzt wird.
        rows = []
        i = 0          # Anker-Frame-Index im Input
        out_idx = 0    # Output-Frame-Zaehler

        while i < n_source_frames:
            hc = int(hold_count[i])
            bw = float(blur_window[i])
            lfps = float(local_fps[i])
            t = float(source_times[i])

            # Blur-Fenster: zentriert um den Anker-Frame.
            # half_window in Frame-Einheiten (kann Bruchteil sein)
            half = bw / 2.0
            blur_start = max(0.0, i - half)
            blur_end   = min(n_source_frames - 1.0, i + half)

            # Erster Frame des Blocks: wird tatsaechlich gerendert
            rows.append({
                "output_frame_idx":    out_idx,
                "output_time_sec":     t,        # == source_time_sec im motion_grade-Modus
                "source_center_frame": i,
                "source_time_sec":     t,
                "blur_start_frame":    blur_start,
                "blur_end_frame":      blur_end,
                "blur_window_frames":  bw,
                "local_target_fps":    lfps,
                "hold_count":          hc,
                "is_hold_frame":       False,
            })
            out_idx += 1

            # Wiederholungs-Frames des Blocks (is_hold_frame=True):
            # kein neuer Render noetig, nur kopieren
            for _ in range(hc - 1):
                rows.append({
                    "output_frame_idx":    out_idx,
                    "output_time_sec":     t,
                    "source_center_frame": i,
                    "source_time_sec":     t,
                    "blur_start_frame":    blur_start,
                    "blur_end_frame":      blur_end,
                    "blur_window_frames":  bw,
                    "local_target_fps":    lfps,
                    "hold_count":          hc,
                    "is_hold_frame":       True,
                })
                out_idx += 1

            # Naechster Anker-Frame: um hold_count springen, damit
            # kein Input-Frame doppelt als Anker genutzt wird.
            i += hc

        df = pd.DataFrame(rows)
        df["mode"] = "motion_grade"
        return df

    def save(self, df: pd.DataFrame, path: str) -> None:
        df.to_csv(path, index=False)
"""
ramp_plotting.py
-------------------
Modus-bewusstes Visualisierungsmodul. Erkennt automatisch ob die
timing_table im speedramp- oder motion_grade-Format vorliegt und
waehlt die passende Darstellung.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _get_mode(df: pd.DataFrame) -> str:
    return df["mode"].iloc[0] if "mode" in df.columns else "speedramp"


def _anchor_frames(df: pd.DataFrame) -> pd.DataFrame:
    """Gibt nur Anker-Frames zurueck (keine Hold-Wiederholungen).
    Im speedramp-Modus ist jeder Frame ein Anker."""
    if "is_hold_frame" in df.columns:
        return df[~df["is_hold_frame"]].copy()
    return df


def plot_frame_mapping(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Panel 1: Zeigt wie Quell-Frames auf Output-Frames abgebildet werden.

    speedramp:    fraktionaler Quell-Frame-Index (source_floor + timestep)
                  gegen output_frame_idx. Steigung < 1 = Zeitlupe.

    motion_grade: source_center_frame gegen output_frame_idx.
                  Im echten 1:1-Modus ist die Steigung immer exakt 1
                  (jeder Output-Frame hat einen eigenen Quell-Anker).
                  Stattdessen zeigen wir blur_window_frames als
                  Fuellung um die 1:1-Linie -- das macht den
                  Blur-Charakter der Rampe direkt sichtbar.
    """
    mode = _get_mode(df)

    if mode == "speedramp":
        source_frame_float = df["source_frame_floor"].values + df["rife_timestep"].values
        ax.plot(df["output_frame_idx"], source_frame_float,
                color="#2563eb", linewidth=1.8)
        ax.set_ylabel("Quell-Frame-Index (fraktional)")
        ax.annotate(
            "flache Steigung = Zeitlupe  |  steile Steigung = Echtzeit/Raffung",
            xy=(0.02, 0.95), xycoords="axes fraction",
            fontsize=8, color="#555555", va="top",
        )
    else:
        anchors = _anchor_frames(df)
        x = anchors["output_frame_idx"].values
        center = anchors["source_center_frame"].values
        half_w = anchors["blur_window_frames"].values / 2.0

        ax.fill_between(x, center - half_w, center + half_w,
                        alpha=0.25, color="#2563eb", label="Blur-Fenster")
        ax.plot(x, center, color="#2563eb", linewidth=1.8, label="Anker-Frame")
        # 1:1-Referenzlinie
        ax.plot([x[0], x[-1]], [x[0], x[-1]],
                color="#94a3b8", linewidth=0.8, linestyle="--", label="1:1 Referenz")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylabel("Quell-Frame-Index")
        ax.annotate(
            "Breite der blauen Fläche = Blur-Fenster  |  bei 100fps-Charakter → schmal",
            xy=(0.02, 0.95), xycoords="axes fraction",
            fontsize=8, color="#555555", va="top",
        )

    ax.set_xlabel("Output-Frame-Index")
    ax.set_title(f"1) Frame-Mapping  [{mode}]")
    ax.grid(True, alpha=0.3)


def plot_effective_framerate(fig: plt.Figure, axes: list[plt.Axes], df: pd.DataFrame) -> None:
    """
    Panel 2a–2d: fps-Kurve und deren Ableitungen.
    Im motion_grade-Modus werden Hold-Frames vor der Ableitung
    herausgefiltert (identische Zeitstempel → dt=0 → Singularitäten).
    """
    mode = _get_mode(df)

    # Nur Anker-Frames fuer Ableitungsberechnung
    work = _anchor_frames(df)
    t   = work["output_time_sec"].values
    fps = work["local_target_fps"].values

    dt = np.diff(t)
    dt = np.where(dt == 0, 1e-9, dt)
    first_deriv = np.diff(fps) / dt
    t_first = t[:-1] + dt / 2.0

    dt2 = np.diff(t_first)
    dt2 = np.where(dt2 == 0, 1e-9, dt2)
    second_deriv = np.diff(first_deriv) / dt2
    t_second = t_first[:-1] + dt2 / 2.0

    dt3 = np.diff(t_second)
    dt3 = np.where(dt3 == 0, 1e-9, dt3)
    if len(second_deriv) > 1:
        third_deriv = np.diff(second_deriv) / dt3
        t_third = t_second[:-1] + dt3 / 2.0
    else:
        third_deriv = np.array([])
        t_third = np.array([])

    ax_fps, ax_d1, ax_d2, ax_d3 = axes

    ax_fps.plot(t, fps, color="#16a34a", linewidth=1.8)
    ax_fps.set_ylabel("lokale Ziel-FPS")
    ax_fps.set_title(f"2a) Effective Framerate Curve  [{mode}]")
    ax_fps.grid(True, alpha=0.3)

    ax_d1.plot(t_first, first_deriv, color="#d97706", linewidth=1.2)
    ax_d1.set_ylabel("d(fps)/dt")
    ax_d1.set_title("2b) Erste Ableitung")
    ax_d1.grid(True, alpha=0.3)
    ax_d1.axhline(0, color="#999999", linewidth=0.8)

    ax_d2.plot(t_second, second_deriv, color="#dc2626", linewidth=1.0)
    ax_d2.set_ylabel("d²(fps)/dt²")
    ax_d2.set_title("2c) Zweite Ableitung — darf hoch sein, muss aber glatt sein")
    ax_d2.grid(True, alpha=0.3)
    ax_d2.axhline(0, color="#999999", linewidth=0.8)

    if len(third_deriv) > 0:
        ax_d3.plot(t_third, third_deriv, color="#9333ea", linewidth=1.0)
        abs_third = np.abs(third_deriv)
        threshold = 25.0 if mode == "motion_grade" else 8.0
        local_median = np.median(abs_third) if len(abs_third) > 0 else 0.0
        spike_mask = abs_third > threshold * local_median
        if np.any(spike_mask):
            ax_d3.scatter(t_third[spike_mask], third_deriv[spike_mask],
                          color="#dc2626", s=30, zorder=5, label="Sprung-Verdacht")
            ax_d3.legend(loc="upper right", fontsize=8)
    ax_d3.set_ylabel("d³(fps)/dt³")
    ax_d3.set_xlabel("Output-Zeit (s)")
    ax_d3.set_title("2d) Ruck — STUFENLOSIGKEITS-BEWEIS (rote Punkte = echter Sprung)")
    ax_d3.grid(True, alpha=0.3)
    ax_d3.axhline(0, color="#999999", linewidth=0.8)
    ax_d3.annotate(
        f"Spike-Schwelle: {25.0 if mode == 'motion_grade' else 8.0}× Median",
        xy=(0.02, 0.92), xycoords="axes fraction", fontsize=8, color="#555555",
    )


def plot_timestep_sanity(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Panel 3: Im speedramp-Modus: rife_timestep Sägezahn-Check.
    Im motion_grade-Modus: blur_window_frames über die Zeit --
    zeigt direkt, wie sich die simulierte Shutter-Breite durch
    die Rampe verändert.
    """
    mode = _get_mode(df)

    if mode == "speedramp":
        ax.plot(df["output_frame_idx"], df["rife_timestep"],
                color="#7c3aed", linewidth=1.0)
        ax.axhspan(1.0, 1.05, color="#dc2626", alpha=0.15)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("RIFE-Timestep (Blend-Faktor)")
        ax.set_title("3) Timestep-Sanity-Check (muss strikt in [0, 1) bleiben)")
        ax.annotate(
            "rote Zone = ungültiger Bereich",
            xy=(0.02, 0.92), xycoords="axes fraction", fontsize=8, color="#555555",
        )
    else:
        anchors = _anchor_frames(df)
        ax.plot(anchors["output_frame_idx"], anchors["blur_window_frames"],
                color="#7c3aed", linewidth=1.4)
        ax.axhline(1.0, color="#16a34a", linewidth=0.8, linestyle="--",
                   label="1.0 = 100fps-Charakter (kein Blur)")
        ax.axhline(4.0, color="#d97706", linewidth=0.8, linestyle="--",
                   label="4.0 = 25fps-Charakter (max. Blur bei 100fps Quelle)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylabel("Blur-Fensterbreite (Quell-Frames)")
        ax.set_title("3) Blur-Fenster-Verlauf  [motion_grade]")
        ax.annotate(
            "Breite = Anzahl der in jeden Output-Frame gefalteten Quell-Frames",
            xy=(0.02, 0.92), xycoords="axes fraction", fontsize=8, color="#555555",
        )

    ax.set_xlabel("Output-Frame-Index")
    ax.grid(True, alpha=0.3)


def build_full_verification_figure(df: pd.DataFrame, title: str = "Ramp-Verifikation") -> plt.Figure:
    fig = plt.figure(figsize=(11, 16))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(6, 1, height_ratios=[2, 1.1, 1.1, 1.1, 1.1, 1.5], hspace=0.6)

    plot_frame_mapping(fig.add_subplot(gs[0]), df)
    plot_effective_framerate(fig,
        [fig.add_subplot(gs[i]) for i in range(1, 5)], df)
    plot_timestep_sanity(fig.add_subplot(gs[5]), df)
    return fig


def save_and_show(fig: plt.Figure, output_path: str | None = None, show: bool = True) -> None:
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Verifikations-Plot gespeichert: {output_path}")
    if show:
        plt.show()
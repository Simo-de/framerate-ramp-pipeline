"""
motion_blur.py
----------------
Sub-Frame-Blur-Compositing fuer Motion Grading.

Im motion_grade-Modus liest dieser Compositor die neuen Spalten
blur_start_frame / blur_end_frame / blur_window_frames aus der
timing_table und samplet gleichmaessig verteilt innerhalb dieses
Fensters -- direkt in Input-Frame-Einheiten, ohne Umweg ueber
Output-Zeitachse. Das ist praeziser und einfacher als die vorherige
velocity_factor-Umrechnung.

Fuer is_hold_frame==True Zeilen: kein RIFE-Aufruf, der zuletzt
gerenderte Frame wird einfach kopiert (spart Rechenzeit proportional
zum hold_count).
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm


class SubFrameBlurCompositor:
    def __init__(
        self,
        source_fps: float,
        output_fps: float,
        shutter_angle: float = 360.0,
        max_subsamples: int = 7,
        mode: str = "motion_grade",
    ):
        if mode not in ("speedramp", "motion_grade"):
            raise ValueError(f"Unbekannter Modus: '{mode}'.")
        self.source_fps    = source_fps
        self.output_fps    = output_fps
        self.shutter_angle = shutter_angle
        self.max_subsamples = max_subsamples
        self.mode          = mode

    # ------------------------------------------------------------------
    # Interne Hilfsmethoden
    # ------------------------------------------------------------------

    def _resolve_source_sample(
        self, frame_float: float, max_source_frame_idx: int
    ) -> tuple[int, int, float]:
        """
        Wandelt einen fraktionalen Quell-Frame-Index (z.B. 12.73) in
        (floor_idx, ceil_idx, timestep) um -- direkt in Frame-Einheiten,
        keine Sekunden-Umrechnung mehr noetig.
        """
        frame_float = float(np.clip(frame_float, 0.0, max_source_frame_idx))
        floor_idx   = int(np.floor(frame_float))
        floor_idx   = int(np.clip(floor_idx, 0, max_source_frame_idx))
        ceil_idx    = int(np.clip(floor_idx + 1, 0, max_source_frame_idx))
        timestep    = frame_float - floor_idx
        if floor_idx == ceil_idx:
            timestep = 0.0
        return floor_idx, ceil_idx, float(np.clip(timestep, 0.0, 1.0))

    def _compute_subsamples_from_window(self, blur_window_frames: float) -> int:
        """
        Anzahl der Sub-Samples aus der Blur-Fensterbreite (in Frames).
        Mindestens 2, hoechstens max_subsamples.
        """
        shutter_fraction = self.shutter_angle / 360.0
        effective_window = blur_window_frames * shutter_fraction
        n = int(np.ceil(max(effective_window, 1.0))) + 1
        return int(np.clip(n, 2, self.max_subsamples))

    # ------------------------------------------------------------------
    # Oeffentliche API
    # ------------------------------------------------------------------

    def render_composite(
        self,
        blur_start_frame: float,
        blur_end_frame: float,
        blur_window_frames: float,
        interpolator,
        source_frames_dir: Path,
        max_source_frame_idx: int,
        frame_filename_pattern: str = "frame_{:06d}.png",
    ) -> np.ndarray:
        """
        Erzeugt einen Blur-Composite aus dem Input-Frame-Fenster
        [blur_start_frame, blur_end_frame].

        Die Sub-Samples werden gleichmaessig innerhalb des durch
        shutter_angle skalierten Fensters verteilt. Bei shutter_angle=360
        deckt das Fenster den vollen blur_window ab; bei 180 nur die
        Haelfte (filmischerer Look).
        """
        shutter_fraction = self.shutter_angle / 360.0
        center = (blur_start_frame + blur_end_frame) / 2.0
        half   = (blur_end_frame - blur_start_frame) / 2.0 * shutter_fraction

        n_samples = self._compute_subsamples_from_window(blur_window_frames)
        sample_positions = np.linspace(center - half, center + half, n_samples)

        accum = None
        for pos in sample_positions:
            floor_idx, ceil_idx, timestep = self._resolve_source_sample(
                pos, max_source_frame_idx
            )
            frame_a = source_frames_dir / frame_filename_pattern.format(floor_idx)
            frame_b = source_frames_dir / frame_filename_pattern.format(ceil_idx)

            sample = interpolator.interpolate_frame(
                str(frame_a), str(frame_b), timestep
            ).astype(np.float32)
            accum = sample if accum is None else accum + sample

        return (accum / n_samples).round().astype(np.uint8)

    def process_timing_table(
        self,
        timing_df: pd.DataFrame,
        interpolator,
        source_frames_dir: str,
        output_dir: str,
        max_source_frame_idx: int,
        frame_filename_pattern: str = "frame_{:06d}.png",
    ) -> None:
        """
        Iteriert ueber die timing_table und schreibt alle Output-Frames.

        Optimierung fuer Hold-Frames (is_hold_frame==True):
        Anstatt RIFE erneut aufzurufen, wird das zuletzt gerenderte
        Bild einfach kopiert. Das spart bei einem 4x-Hold-Count 75%
        der RIFE-Forward-Passes in den 25fps-Bereichen der Rampe.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        src_path = Path(source_frames_dir)

        last_rendered: np.ndarray | None = None
        skipped_existing = 0
        rendered = 0
        held = 0

        mode = timing_df["mode"].iloc[0] if "mode" in timing_df.columns else "motion_grade"

        for _, row in tqdm(timing_df.iterrows(), total=len(timing_df),
                           desc="Motion-Grade-Compositing"):
            out_name = frame_filename_pattern.format(int(row["output_frame_idx"]))
            out_file = output_path / out_name

            if out_file.exists():
                skipped_existing += 1
                # last_rendered aktualisieren, falls wir mitten in einem
                # laufenden Hold-Block weitermachen (Resume-Fall)
                if last_rendered is None:
                    last_rendered = cv2.imread(str(out_file))
                continue

            if mode == "motion_grade" and row.get("is_hold_frame", False):
                # Hold-Frame: letzten Composite wiederverwenden
                if last_rendered is not None:
                    cv2.imwrite(str(out_file), last_rendered)
                    held += 1
                    continue

            # Echter Render-Frame
            if mode == "motion_grade":
                result = self.render_composite(
                    blur_start_frame   = float(row["blur_start_frame"]),
                    blur_end_frame     = float(row["blur_end_frame"]),
                    blur_window_frames = float(row["blur_window_frames"]),
                    interpolator       = interpolator,
                    source_frames_dir  = src_path,
                    max_source_frame_idx = max_source_frame_idx,
                    frame_filename_pattern = frame_filename_pattern,
                )
            else:
                # Speedramp-Fallback: altes Verhalten (Einzelframe-Sample)
                frame_a = src_path / frame_filename_pattern.format(
                    int(row["source_frame_floor"]))
                frame_b = src_path / frame_filename_pattern.format(
                    int(row["source_frame_ceil"]))
                result = interpolator.interpolate_frame(
                    str(frame_a), str(frame_b), float(row["rife_timestep"])
                )

            cv2.imwrite(str(out_file), result)
            last_rendered = result
            rendered += 1

        total = rendered + held
        if total > 0:
            efficiency = held / total * 100
            print(f"[INFO] Gerendert: {rendered} Composites, "
                  f"gehalten: {held} Hold-Frames ({efficiency:.1f}% RIFE-Ersparnis).")
        if skipped_existing:
            print(f"[INFO] {skipped_existing} vorhandene Frames übersprungen (Resume).")
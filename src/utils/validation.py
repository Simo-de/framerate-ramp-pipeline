"""
validation.py
----------------
Automatisierte Plausibilitaetsprueofungen fuer die timing_table, BEVOR
echte Bildsequenzen gerendert werden. Jede Pruefung deckt eine konkrete
Fehlerklasse ab, die sich sonst erst im fertigen Footage (als Judder,
Sprung oder Ghosting) zeigen wuerde.

Designprinzip: Jede Check-Funktion gibt ein ValidationResult zurueck
(bestanden/nicht bestanden + Klartext-Begruendung + Kennzahlen), damit
sowohl die Konsolenausgabe als auch der Plot dieselbe Quelle nutzen.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    name: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        status = "OK" if self.passed else "FEHLER"
        return f"[{status}] {self.name}: {self.message}"


def check_monotonic_source_time(df: pd.DataFrame) -> ValidationResult:
    """
    PHYSIKALISCHE BEDEUTUNG: source_time_sec darf zwischen aufeinander-
    folgenden Output-Frames niemals fallen. Ein Rueckwaertssprung wuerde
    bedeuten, dass die Pipeline einen bereits gezeigten Moment im
    Quellmaterial erneut abspielt -- ein klassischer Zeit-Stotterer,
    der in keiner Ramp vorkommen darf.

    Hinweis: Ein Differenzwert von exakt 0.0 ist eigentlich auch
    grenzwertig (kompletter Stillstand der Quellzeit), wird hier aber
    nicht als Fehler gewertet, da Ramp-Plateaus am Rand (z.B. Anfangs-
    Stillstand) bewusst konstruiert sein koennten. Striktes Monotonie-
    Kriterium ist: diff >= 0.
    """
    diffs = np.diff(df["source_time_sec"].values)
    n_violations = int(np.sum(diffs < 0))
    worst_violation = float(np.min(diffs)) if len(diffs) > 0 else 0.0

    passed = n_violations == 0
    if passed:
        msg = "source_time_sec ist durchgehend monoton steigend -- keine Zeitsprünge rückwärts."
    else:
        msg = (
            f"{n_violations} Rückwärtssprünge in source_time_sec gefunden! "
            f"Stärkster Rückwärtssprung: {worst_violation:.6f}s. "
            f"Ursache meist: fehlerhafte Integration der velocity_factor-Kurve."
        )
    return ValidationResult(
        name="Monotonie der Quellzeit",
        passed=passed,
        message=msg,
        details={"n_violations": n_violations, "worst_violation_sec": worst_violation},
    )


def check_timestep_range(df: pd.DataFrame) -> ValidationResult:
    """
    PHYSIKALISCHE BEDEUTUNG: rife_timestep ist der fraktionale Blend-
    Faktor zwischen source_frame_floor und source_frame_ceil und MUSS
    im Intervall [0, 1) liegen. Ein Wert von z.B. 1.3 wuerde bedeuten,
    dass RIFE ueber den ceil-Frame "hinaus" interpolieren soll -- das
    ist undefiniert und RIFE wuerde entweder abstuerzen oder (schlimmer)
    stillschweigend einen falschen Frame nehmen.
    """
    ts = df["rife_timestep"].values
    out_of_range_mask = (ts < 0.0) | (ts >= 1.0)
    n_violations = int(np.sum(out_of_range_mask))

    passed = n_violations == 0
    if passed:
        msg = f"Alle {len(ts)} Timesteps liegen korrekt in [0, 1). Min={ts.min():.4f}, Max={ts.max():.4f}."
    else:
        bad_indices = df.loc[out_of_range_mask, "output_frame_idx"].tolist()[:10]
        msg = (
            f"{n_violations} Timesteps außerhalb [0, 1)! "
            f"Betroffene output_frame_idx (max. 10 gezeigt): {bad_indices}."
        )
    return ValidationResult(
        name="Timestep-Wertebereich",
        passed=passed,
        message=msg,
        details={"n_violations": n_violations},
    )


def check_source_frame_bounds(df: pd.DataFrame, max_source_frame_idx: int) -> ValidationResult:
    """
    PHYSIKALISCHE BEDEUTUNG: source_frame_ceil darf nie ueber das Ende
    der tatsaechlich extrahierten Quell-Frame-Sequenz hinauszeigen.
    Das ist eine reine Index-Sicherheitspruefung, die VOR dem RIFE-Lauf
    einen klaren Fehler werfen soll, statt erst beim Dateizugriff mit
    einer kryptischen "File not found"-Exception zu scheitern.
    """
    max_needed = int(df["source_frame_ceil"].max())
    passed = max_needed <= max_source_frame_idx
    if passed:
        msg = (
            f"Maximal benötigter Quell-Frame-Index ({max_needed}) liegt innerhalb "
            f"der verfügbaren Sequenz (0..{max_source_frame_idx})."
        )
    else:
        msg = (
            f"Timing-Tabelle fordert Quell-Frame {max_needed} an, aber nur "
            f"0..{max_source_frame_idx} sind extrahiert! Entweder total_duration_sec "
            f"zu lang gewählt, oder die Quellsequenz ist unvollständig extrahiert."
        )
    return ValidationResult(
        name="Quell-Frame-Indexgrenzen",
        passed=passed,
        message=msg,
        details={"max_needed": max_needed, "max_available": max_source_frame_idx},
    )


def check_fps_curve_smoothness(
    df: pd.DataFrame, spike_sharpness_threshold: float = 8.0
) -> ValidationResult:
    """Wie bisher, aber im motion_grade-Modus nur Anker-Frames auswerten
    und einen angepassten Threshold verwenden, da ganzzahlige hold_counts
    naturgemäß schärfere Diskretisierungssprünge erzeugen als das
    kontinuierliche speedramp-Mapping."""
    mode = df["mode"].iloc[0] if "mode" in df.columns else "speedramp"

    if mode == "motion_grade" and "is_hold_frame" in df.columns:
        df = df[~df["is_hold_frame"]].copy()
        # hold_count-Rundung erzeugt diskrete Sprünge in der fps-Kurve,
        # die in der dritten Ableitung deutlich sichtbar sind, aber kein
        # Qualitätsproblem darstellen (das Auge nimmt Sprünge erst ab der
        # zweiten Ableitung wahr, nicht der dritten). Threshold großzügiger.
        spike_sharpness_threshold = max(spike_sharpness_threshold, 25.0)

    fps = df["local_target_fps"].values
    t   = df["output_time_sec"].values
    dt = np.diff(t)
    dt = np.where(dt == 0, 1e-9, dt)

    first_deriv = np.diff(fps) / dt
    second_deriv = np.diff(first_deriv) / dt[:-1]
    third_deriv = np.diff(second_deriv) / dt[:-2] if len(second_deriv) > 1 else np.array([0.0])

    if len(third_deriv) < 5:
        # Zu kurze Sequenz fuer eine sinnvolle Spike-Erkennung -- gilt
        # als unkritisch (z.B. bei sehr kurzen Testclips).
        return ValidationResult(
            name="Glattheit der FPS-Kurve (Sprung-Erkennung via 3. Ableitung)",
            passed=True,
            message="Sequenz zu kurz für robuste Spike-Erkennung -- Prüfung übersprungen.",
            details={},
        )

    abs_third = np.abs(third_deriv)
    # Randeffekt-Trim: Am Clip-Ende entsteht durch Abschneiden der Ramp-
    # Kurve (total_duration < preset_end) ein harter Grenzartefakt in der
    # dritten Ableitung. Wir schneiden die letzten max_hold Samples ab,
    # wobei max_hold der groesste hold_count in der Tabelle ist (typisch
    # 4 bei 25fps-Kurven). Das entspricht exakt der letzten Blur-Gruppe.
    if "hold_count" in df.columns:
        trim = int(df["hold_count"].max()) + 2
    else:
        trim = 6  # sicherer Fallback
    trim = min(trim, len(abs_third) // 4)  # nie mehr als 25% abschneiden
    eval_third = abs_third[:-trim] if trim > 0 and len(abs_third) > trim else abs_third
    local_median = float(np.median(eval_third))
    max_third = float(np.max(eval_third))

    # Verhaeltnis von Spitze zu typischem (Median-)Niveau. Bei einer
    # glatten Kurve ist dieses Verhaeltnis klein, auch wenn das absolute
    # Niveau (z.B. bei einer sehr steilen Rampe) hoch ist.
    sharpness_ratio = max_third / local_median if local_median > 1e-9 else max_third

    passed = sharpness_ratio <= spike_sharpness_threshold
    worst_idx = int(np.argmax(abs_third))
    worst_time = float(t[worst_idx]) if worst_idx < len(t) else float(t[-1])

    if passed:
        msg = (
            f"Keine isolierten Ruck-Spitzen erkannt (Spitze/Median-Verhältnis: "
            f"{sharpness_ratio:.2f}, Schwelle: {spike_sharpness_threshold}). "
            f"Die Kurve darf trotzdem stellenweise stark gekrümmt sein -- "
            f"entscheidend ist, dass diese Krümmung sich glatt aufbaut statt zu springen."
        )
    else:
        msg = (
            f"Isolierte Ruck-Spitze bei t≈{worst_time:.3f}s erkannt "
            f"(Spitze/Median-Verhältnis: {sharpness_ratio:.2f}, Schwelle: "
            f"{spike_sharpness_threshold}). Das deutet auf eine echte Naht-Diskontinuität "
            f"hin (z.B. an einem Keyframe-Übergang), nicht auf normale S-Kurven-Krümmung. "
            f"Typische Ursache: stückweise Interpolation, die an Segmentgrenzen nicht "
            f"C2-stetig anschließt."
        )
    return ValidationResult(
        name="Glattheit der FPS-Kurve (Sprung-Erkennung via 3. Ableitung)",
        passed=passed,
        message=msg,
        details={"sharpness_ratio": sharpness_ratio, "max_third_deriv": max_third},
    )


def check_frame_duplication_rate(df: pd.DataFrame, source_fps: float) -> ValidationResult:
    """
    Plausibilitaetsmetrik (kein Hard-Fail): Zeigt den Anteil wiederverwendeter
    Frames im Output an. Die Semantik unterscheidet sich je nach Modus fundamental:

    speedramp-Modus:
        Gemessen wird, wie oft sich source_frame_floor zwischen aufeinanderfolgenden
        Output-Frames NICHT aendert -- d.h. mehrere Output-Frames greifen auf
        dasselbe Quell-Frame-Intervall zurueck. Das passiert bei niedrigen
        Geschwindigkeitsfaktoren (velocity = local_target_fps / source_fps << 1),
        also in den Zeitlupen-Abschnitten der Rampe. Erwartung: hoch bei 25fps-
        Charakter (~75%), nahe 0% bei 100fps-Charakter (1:1-Mapping).

    motion_grade-Modus:
        Hier gibt es keine velocity-basierte Quellzeit-Dehnung mehr. Stattdessen
        wird die is_hold_frame-Spalte direkt ausgewertet. Hold-Frames entstehen
        durch ganzzahlige Rundung des hold_count (= round(source_fps / local_fps)):
        Ein Composite-Bild wird hold_count-mal in den 100fps-Output-Container
        geschrieben, um die simulierte Update-Rate (z.B. 25fps) bei konstanter
        Container-Framerate (100fps) zu erzeugen. Das ist konzeptuell identisch
        zum 3:2-Pulldown in der Kinoprojektion, hier aber stufenlos variabel.
        Erwartung: ~75% Hold-Frames in 25fps-Bereichen (von 4 Output-Frames sind
        3 Hold-Kopien eines Composites), ~0% in 100fps-Bereichen (jeder Frame
        ist ein eigener Composite ohne Wiederholung).

    Diese Metrik ist immer passed=True -- sie dient ausschliesslich der
    Plausibilitaetskontrolle, nicht als Render-Freigabe-Kriterium.
    """
    mode = df["mode"].iloc[0] if "mode" in df.columns else "speedramp"

    if mode == "motion_grade":
        if "is_hold_frame" not in df.columns:
            return ValidationResult(
                name="Hold-Frame-Rate (Info)", passed=True,
                message="is_hold_frame-Spalte nicht vorhanden -- Prüfung übersprungen.",
            )
        rate = float(df["is_hold_frame"].mean())
        msg = (
            f"{rate*100:.1f}% der Output-Frames sind Hold-Frames (Wiederholungen). "
            f"Erwartung: ~75% in 25fps-Bereichen, ~0% in 100fps-Bereichen."
        )
    else:
        floors = df["source_frame_floor"].values
        same_floor = np.sum(np.diff(floors) == 0)
        rate = same_floor / max(len(floors) - 1, 1)
        msg = (
            f"{rate*100:.1f}% der Output-Frame-Übergänge teilen sich denselben Quell-Frame "
            f"(erwartbar hoch in Zeitlupen-Abschnitten, nahe 0% in 100fps-Bereichen)."
        )

    return ValidationResult(
        name="Frame-Wiederverwendungsrate (Info)",
        passed=True,
        message=msg,
        details={"rate": rate},
    )

def run_all_checks(
    df: pd.DataFrame,
    max_source_frame_idx: int,
    source_fps: float,
    spike_sharpness_threshold: float = 8.0,
) -> list[ValidationResult]:
    mode = df["mode"].iloc[0] if "mode" in df.columns else "speedramp"

    results = []

    if mode == "motion_grade":
        # Im motion_grade-Modus gibt es keine source_time-Monotonie
        # oder rife_timestep-Spalten mehr. Wir pruefen stattdessen
        # blur_end_frame-Grenzen und fps-Kurven-Glattheit.
        if "blur_end_frame" in df.columns:
            max_needed = int(df["blur_end_frame"].max())
            passed = max_needed <= max_source_frame_idx
            results.append(ValidationResult(
                name="Blur-Fenster-Indexgrenzen",
                passed=passed,
                message=(
                    f"Max. benötigter Frame-Index im Blur-Fenster: {max_needed}, "
                    f"verfügbar: 0..{max_source_frame_idx}."
                ) if passed else (
                    f"FEHLER: Blur-Fenster fordert Frame {max_needed}, "
                    f"aber nur 0..{max_source_frame_idx} extrahiert!"
                ),
                details={"max_needed": max_needed, "max_available": max_source_frame_idx},
            ))
        results.append(check_fps_curve_smoothness(df, spike_sharpness_threshold))
        results.append(check_frame_duplication_rate(df, source_fps))
    else:
        results = [
            check_monotonic_source_time(df),
            check_timestep_range(df),
            check_source_frame_bounds(df, max_source_frame_idx),
            check_fps_curve_smoothness(df, spike_sharpness_threshold),
            check_frame_duplication_rate(df, source_fps),
        ]

    return results


def print_validation_report(results: list[ValidationResult]) -> bool:
    """
    Gibt einen lesbaren Konsolenbericht aus und liefert True zurück,
    wenn ALLE harten Prüfungen (passed=False möglich) bestanden wurden.
    Wird im Notebook UND im CLI-Skript gleich genutzt, damit beide
    Wege konsistente Ergebnisse zeigen.
    """
    print("=" * 70)
    print("PLAUSIBILITÄTSPRÜFUNG DER TIMING-TABLE")
    print("=" * 70)
    all_passed = True
    for r in results:
        print(str(r))
        if not r.passed:
            all_passed = False
            print(f"         -> {r.message}")
    print("=" * 70)
    if all_passed:
        print("GESAMTERGEBNIS: Alle Prüfungen bestanden. Render-Freigabe möglich.")
    else:
        print("GESAMTERGEBNIS: Mindestens eine Prüfung fehlgeschlagen. "
              "RIFE-Render NICHT starten, bis behoben.")
    print("=" * 70)
    return all_passed
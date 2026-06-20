"""
pipeline.py
-------------
CLI-Entry-Point: orchestriert den gesamten Ablauf von der Ramp-Definition
bis zum RIFE-interpolierten Frame-Output.

Aufruf-Beispiel (voller Durchlauf inkl. RIFE):
    python -m src.pipeline --clip clip_001 --preset 25_100_25_smoothstep \
        --duration 4.5 --rife-model-dir third_party/practical-rife/train_log

Aufruf-Beispiel (nur Timing-Tabelle, RIFE überspringen -- z.B. zur
erneuten Verifikation nach einer Preset-Änderung):
    python -m src.pipeline --clip clip_001 --preset 25_100_25_smoothstep \
        --skip-rife
"""

from __future__ import annotations
import argparse
import json
import re
import shutil
from pathlib import Path

from src.ramp_curve import RampCurve
from src.timing_table import TimingTableBuilder
from src.utils.validation import run_all_checks, print_validation_report
from src.utils.ramp_plotting import build_full_verification_figure, save_and_show


def _render_fingerprint(args: argparse.Namespace, n_output_frames: int) -> dict:
    """
    Erzeugt einen Fingerprint der aktuellen Render-Konfiguration.
    Alle Parameter, die das Aussehen der gerenderten Frames beeinflussen,
    fliessen hier ein. Parameter die NICHT einfliessen: --rife-model-dir
    (Pfad, nicht Inhalt), --force (Meta-Flag), --skip-rife/--skip-blur.
    """
    return {
        "clip":            args.clip,
        "preset":          args.preset,
        "mode":            args.mode,
        "source_fps":      args.source_fps,
        "output_fps":      args.output_fps,
        "shutter_angle":   args.shutter_angle,
        "max_subsamples":  args.max_subsamples,
        "scale":           args.scale,
        "n_output_frames": n_output_frames,
    }


def _fingerprint_path(render_dir: Path) -> Path:
    return render_dir / ".render_fingerprint.json"


def _save_fingerprint(render_dir: Path, fingerprint: dict) -> None:
    fp_path = _fingerprint_path(render_dir)
    with open(fp_path, "w", encoding="utf-8") as f:
        json.dump(fingerprint, f, indent=2)


def _check_cache_compatibility(
    render_dir: Path, current_fp: dict, force: bool
) -> bool:
    """
    Vergleicht den vorhandenen Cache-Fingerprint mit der aktuellen
    Konfiguration. Gibt True zurück wenn Resume sicher ist, False wenn
    der Cache gelöscht werden muss.

    Verhalten:
      - Kein Fingerprint vorhanden (alter Cache ohne diese Funktion):
        Warnung + SystemExit. Der Nutzer muss explizit --force setzen
        oder manuell löschen.
      - Fingerprint vorhanden und kompatibel: Resume erlaubt.
      - Fingerprint vorhanden aber inkompatibel: Warnung + SystemExit,
        ausser --force ist gesetzt (dann: löschen + neu rendern).
    """
    fp_path = _fingerprint_path(render_dir)
    existing_frames = list(render_dir.glob("frame_*.png"))

    if not existing_frames:
        # Leeres Verzeichnis: kein Resume-Konflikt moeglich
        return True

    if not fp_path.exists():
        # Frames ohne Fingerprint = alter Cache aus Zeit vor diesem Feature
        if force:
            print(f"[FORCE] Kein Fingerprint gefunden -- lösche {len(existing_frames)} "
                  f"vorhandene Frames in {render_dir} und starte neu.")
            shutil.rmtree(render_dir)
            render_dir.mkdir(parents=True)
            return True
        raise SystemExit(
            f"FEHLER: {render_dir} enthält {len(existing_frames)} gerenderte Frames, "
            f"aber keinen Konfigurations-Fingerprint. Das bedeutet, dieser Cache "
            f"stammt aus einer früheren Pipeline-Version oder einer anderen Konfiguration. "
            f"Starte mit --force neu um den Cache zu löschen und sicher neu zu rendern:\n"
            f"  python -m src.pipeline ... --force"
        )

    with open(fp_path, "r", encoding="utf-8") as f:
        cached_fp = json.load(f)

    # Nur die Schlüssel vergleichen, die im aktuellen Fingerprint vorhanden sind,
    # damit zukünftige neue Schlüssel keinen falschen Inkompatibilitäts-Alarm auslösen.
    mismatches = {
        k: (cached_fp.get(k), current_fp[k])
        for k in current_fp
        if cached_fp.get(k) != current_fp[k]
    }

    if not mismatches:
        print(f"[OK] Cache-Fingerprint kompatibel -- Resume aktiv "
              f"({len(existing_frames)} vorhandene Frames werden übersprungen).")
        return True

    mismatch_str = "\n".join(
        f"    {k}: war={v[0]!r}  jetzt={v[1]!r}"
        for k, v in mismatches.items()
    )
    if force:
        print(f"[FORCE] Inkompatible Parameter gegenüber vorhandenem Cache:\n"
              f"{mismatch_str}\n"
              f"  Lösche {len(existing_frames)} Frames und starte neu.")
        shutil.rmtree(render_dir)
        render_dir.mkdir(parents=True)
        return True

    raise SystemExit(
        f"FEHLER: Vorhandener Frame-Cache in {render_dir} wurde mit einer "
        f"anderen Konfiguration gerendert:\n{mismatch_str}\n\n"
        f"Das würde zu einem gemischten Output führen (genau das Problem, "
        f"das du gerade hattest). Optionen:\n"
        f"  1. --force hinzufügen: löscht den Cache automatisch und rendert neu\n"
        f"  2. Manuell löschen: Remove-Item -Recurse -Force \"{render_dir}\"\n"
        f"  3. Anderen --clip oder --preset Namen wählen, um einen neuen "
        f"     Cache-Ordner zu erstellen"
    )


def verify_frame_sequence(source_frames_dir: Path, pattern: str = r"frame_(\d+)\.png") -> int:
    """
    Prüft die tatsächlich extrahierte Quell-Frame-Sequenz auf zwei
    Fehlerklassen, die sonst erst als kryptischer cv2.imwrite-Crash
    MITTEN im RIFE-Lauf auffallen würden (siehe realer Vorfall: FFmpeg
    benennt Frames standardmäßig 1-indexiert -- frame_000001.png statt
    frame_000000.png -- weil das die Default-Konvention des image2-
    Muxers ist, NICHT weil etwas kaputt ist):

      1. Falscher Start-Index: Die Pipeline (timing_table.py) ist
         durchgehend 0-indexiert. Falls die Sequenz tatsächlich bei
         frame_000001.png beginnt, würde jeder Versuch, frame_000000.png
         zu lesen, fehlschlagen -- aber eine reine Anzahl-Zählung
         ("len(files) - 1") würde das NICHT erkennen, da sie nur die
         Dateimenge zählt, nicht den tatsächlichen Indexbereich prüft.
      2. Lücken in der Sequenz: fehlende Einzelframes mitten in der
         Extraktion (z.B. durch einen FFmpeg-Abbruch) würden sonst erst
         beim Zugriff auf den fehlenden Index als Crash auffallen.

    Gibt den höchsten gültigen, lückenlos ab 0 vorhandenen Frame-Index
    zurück (= max_source_frame_idx für die Bounds-Prüfung).
    """
    regex = re.compile(pattern)
    indices = []
    for f in source_frames_dir.glob("frame_*.png"):
        m = regex.match(f.name)
        if m:
            indices.append(int(m.group(1)))

    if not indices:
        raise SystemExit(
            f"Keine extrahierten Frames in {source_frames_dir} gefunden. "
            f"Zuerst mit FFmpeg extrahieren (siehe Schritt 5 der Anleitung)."
        )

    indices.sort()

    if indices[0] != 0:
        raise SystemExit(
            f"FEHLER: Die Frame-Sequenz in {source_frames_dir} beginnt bei "
            f"Index {indices[0]} (erste Datei: frame_{indices[0]:06d}.png), "
            f"nicht bei 0. Das ist die FFmpeg-Standardkonvention -- der "
            f"image2-Muxer startet von Haus aus bei 1, nicht bei 0. Unsere "
            f"Pipeline ist durchgehend 0-indexiert (siehe timing_table.py), "
            f"daher MUSS die Sequenz bei frame_000000.png beginnen.\n\n"
            f"FIX: Frames mit folgendem FFmpeg-Befehl neu extrahieren "
            f"(zusätzliches Flag -start_number 0):\n"
            f"  ffmpeg -i <input> -start_number 0 {source_frames_dir}/frame_%06d.png"
        )

    # Lückenlosigkeit prüfen: jeder Index von 0 bis max muss vorhanden sein.
    expected = set(range(indices[0], indices[-1] + 1))
    actual = set(indices)
    missing = sorted(expected - actual)
    if missing:
        shown = missing[:10]
        raise SystemExit(
            f"FEHLER: {len(missing)} fehlende Frame(s) in der Sequenz "
            f"{source_frames_dir} erkannt. Fehlende Indizes (max. 10 gezeigt): "
            f"{shown}. Extraktion wahrscheinlich unterbrochen oder unvollständig -- "
            f"bitte erneut mit FFmpeg extrahieren."
        )

    print(
        f"[OK] Frame-Sequenz verifiziert: {len(indices)} Frames, lückenlos von "
        f"frame_{indices[0]:06d}.png bis frame_{indices[-1]:06d}.png."
    )
    return indices[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frame Rate Ramp Pipeline")
    parser.add_argument("--clip", required=True, help="Clip-ID, z.B. clip_001")
    parser.add_argument("--preset", required=True, help="Ramp-Preset-Name (ohne .json)")
    parser.add_argument("--source-fps", type=float, default=100.0)
    parser.add_argument("--output-fps", type=float, default=25.0)
    parser.add_argument("--duration", type=float, default=None,
                         help="Gesamtdauer des Clips in Sekunden. Wird weggelassen, "
                              "ermittelt die Pipeline die Dauer automatisch aus der "
                              "Anzahl der extrahierten PNG-Frames in source_100fps/<clip>/frames/. "
                              "Falls angegeben und kürzer als der Clip: wird so verwendet. "
                              "Falls angegeben und länger als der Clip: wird auf Clip-Länge "
                              "gekürzt und eine Warnung ausgegeben.")
    parser.add_argument("--rife-model-dir", type=str, default=None,
                         help="Pfad zum train_log-Verzeichnis von Practical-RIFE. "
                              "Erforderlich, außer --skip-rife ist gesetzt.")
    parser.add_argument("--skip-rife", action="store_true",
                         help="Nur Timing-Tabelle + Verifikation, RIFE-Render überspringen "
                              "(nützlich zum schnellen Iterieren auf der Ramp-Kurve selbst).")
    parser.add_argument("--skip-blur", action="store_true",
                         help="RIFE-Interpolation läuft, aber Blur-Pass überspringen. "
                              "Nützlich um RIFE-Output isoliert zu begutachten, bevor "
                              "Blur hinzugefügt wird.")
    parser.add_argument("--scale", type=float, default=1.0,
                         help="RIFE-interner Skalierungsfaktor. 0.5 für 4K-Material empfohlen.")
    parser.add_argument("--shutter-angle", type=float, default=360.0,
                         help="Virtueller Shutter-Winkel für den Blur-Pass in Grad. "
                              "360=volles Shutter-Integral (Default, passend zum "
                              "Quellmaterial). 180 für filmischeren Look mit weniger Blur.")
    parser.add_argument("--max-subsamples", type=int, default=7,
                         help="Max. RIFE-Sub-Samples pro Output-Frame im Blur-Pass. "
                              "Höher = genaueres Blur-Integral, aber proportional mehr "
                              "Renderzeit. 5-7 ist ein guter Kompromiss.")
    parser.add_argument(
        "--mode",
        choices=["motion_grade", "speedramp"],
        default="motion_grade",
        help=(
            "motion_grade (Default): konstante 1x-Geschwindigkeit, die fps-Kurve "
            "steuert ausschliesslich den Shutter-Look (Motion Grading). "
            "speedramp: fps-Kurve steuert Zeitdehnung, erzeugt Slow Motion / Zeitraffer."
        ),
    )
    parser.add_argument("--output-fps-reassembly", type=float, default=None,
                         help="FPS für den finalen FFmpeg-Reassembly-Schritt. Default: "
                              "identisch zu --output-fps (25). Setze z.B. auf 100, wenn "
                              "du einen Variable-Framerate-Container willst.")
    parser.add_argument("--force", action="store_true",
                         help="Vorhandene gerenderte Frames im Output-Verzeichnis löschen "
                              "und komplett neu rendern. Verwende diesen Flag immer, wenn "
                              "du mode, preset, shutter-angle oder andere Parameter geändert "
                              "hast, seit der letzte Render-Lauf gestartet wurde. Ohne --force "
                              "prüft die Pipeline automatisch, ob der vorhandene Cache mit der "
                              "aktuellen Konfiguration kompatibel ist, und stoppt mit einer "
                              "Warnung wenn nicht.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_rife and not args.rife_model_dir:
        raise SystemExit(
            "Fehler: --rife-model-dir ist erforderlich, außer --skip-rife ist gesetzt. "
            "Beispiel: --rife-model-dir third_party/practical-rife/train_log"
        )

    project_root = Path(__file__).resolve().parent.parent
    preset_path = project_root / "config" / "ramp_presets" / f"{args.preset}.json"
    source_frames_dir = project_root / "data" / "source_100fps" / args.clip / "frames"
    interim_dir = project_root / "data" / "interim" / args.clip / f"ramp_{args.preset}"
    interim_dir.mkdir(parents=True, exist_ok=True)

    # --- Schritt 1: Frame-Sequenz verifizieren und Duration ableiten ---
    # Bewusst VOR der Timing-Table-Berechnung, damit ein falsches
    # --duration nicht erst nach teurer Arbeit auffällt.
    max_source_frame_idx = verify_frame_sequence(source_frames_dir)
    clip_duration_sec = (max_source_frame_idx + 1) / args.source_fps

    if args.duration is None:
        total_duration_sec = clip_duration_sec
        print(f"[INFO] --duration nicht angegeben. Automatisch ermittelt: "
              f"{total_duration_sec:.4f}s ({max_source_frame_idx + 1} Frames @ {args.source_fps:.0f}fps)")
    elif args.duration > clip_duration_sec:
        total_duration_sec = clip_duration_sec
        print(f"[WARNUNG] --duration {args.duration:.4f}s überschreitet Clip-Länge "
              f"({clip_duration_sec:.4f}s). Wird automatisch auf Clip-Länge gekürzt.")
    else:
        total_duration_sec = args.duration
        print(f"[OK] Duration: {total_duration_sec:.4f}s (manuell, "
              f"Clip hat {clip_duration_sec:.4f}s)")

    # --- Schritt 2: Ramp-Kurve laden, Preset-Länge prüfen, Timing-Tabelle bauen ---
    ramp = RampCurve.from_json(str(preset_path))

    # Falls das Preset länger definiert ist als der Clip, informieren wir
    # den Nutzer -- die Kurve wird am Ende des Clips einfach abgeschnitten,
    # was in den meisten Fällen unproblematisch ist (der letzte Keyframe
    # wird nicht mehr erreicht, aber die Kurve ist bis dahin gültig).
    preset_end = ramp.keyframes[-1].time_sec
    if preset_end > total_duration_sec:
        print(f"[INFO] Preset-Dauer ({preset_end:.2f}s) > Clip-Dauer ({total_duration_sec:.4f}s). "
              f"Die Ramp-Kurve wird am Clip-Ende abgeschnitten. Die fps-Kurve "
              f"endet bei {ramp.target_fps(total_duration_sec):.1f}fps statt beim "
              f"letzten Keyframe-Wert von {ramp.keyframes[-1].target_fps:.1f}fps.")

    builder = TimingTableBuilder(
        ramp,
        source_fps=args.source_fps,
        output_fps=args.output_fps,
        mode=args.mode,
    )
    timing_df = builder.build(total_duration_sec=total_duration_sec)
    timing_csv_path = interim_dir / "timing_table.csv"
    builder.save(timing_df, str(timing_csv_path))
    print(f"[OK] Timing-Tabelle gespeichert: {timing_csv_path} "
          f"({len(timing_df)} Output-Frames)")

    # --- Schritt 3: Inhaltliche Plausibilitätsprüfung ---
    # max_source_frame_idx wurde bereits in Schritt 1 ermittelt.
    results = run_all_checks(timing_df, max_source_frame_idx=max_source_frame_idx, source_fps=args.source_fps)
    all_passed = print_validation_report(results)

    fig = build_full_verification_figure(timing_df, title=f"{args.clip} / {args.preset}")
    save_and_show(fig, output_path=str(interim_dir / "verification_plot.png"), show=False)

    if not all_passed:
        raise SystemExit(
            "Pipeline gestoppt: Plausibilitätsprüfung fehlgeschlagen. "
            "Siehe Konsolenausgabe oben und den verification_plot.png. "
            "RIFE-Render wird NICHT gestartet."
        )

    if args.skip_rife:
        print("[INFO] --skip-rife gesetzt: Pipeline endet hier nach erfolgreicher Verifikation.")
        return

    # --- Schritt 4: RIFE-Modell laden ---
    # Lazy-Import, damit --skip-rife auch ohne installiertes Torch/RIFE
    # funktioniert (z.B. zum schnellen Iterieren auf der Ramp-Kurve auf
    # einer Maschine ohne GPU).
    from src.rife_interpolator import RifeInterpolator

    interpolator = RifeInterpolator(model_dir=args.rife_model_dir, scale=args.scale)
    interpolator.load_model()

    if args.skip_blur:
        rife_frames_dir = interim_dir / "rife_frames"
        current_fp = _render_fingerprint(args, n_output_frames=len(timing_df))
        _check_cache_compatibility(rife_frames_dir, current_fp, force=args.force)

        if args.mode == "motion_grade":
            # Im motion_grade-Modus gibt es keine rife_timestep/source_frame_floor-
            # Spalten -- der rohe Interpolator würde hier abstürzen. Wir nutzen
            # stattdessen den Compositor mit shutter_angle=0 (kein Blur, aber
            # korrekte Hold-Frame-Logik) als "RIFE-only"-Äquivalent.
            from src.motion_blur import SubFrameBlurCompositor
            rife_frames_dir = interim_dir / "rife_frames"
            compositor_noblur = SubFrameBlurCompositor(
                source_fps=args.source_fps,
                output_fps=args.output_fps,
                shutter_angle=1.0,   # minimales Fenster ≈ kein Blur
                max_subsamples=2,
                mode=args.mode,
            )
            compositor_noblur.process_timing_table(
                timing_df, interpolator,
                str(source_frames_dir), str(rife_frames_dir),
                max_source_frame_idx=max_source_frame_idx,
            )
        else:
            rife_frames_dir = interim_dir / "rife_frames"
            interpolator.process_timing_table(
                timing_df, str(source_frames_dir), str(rife_frames_dir)
            )
        final_frames_dir = rife_frames_dir
        _save_fingerprint(rife_frames_dir, current_fp)
        print(f"[OK] RIFE-Interpolation abgeschlossen (kein Blur-Pass): {rife_frames_dir}")
    else:
        # --- Schritt 5: Sub-Frame-Blur-Compositing ---
        # Der Blur-Pass erzeugt seine Frames direkt aus dem Quellmaterial
        # (nicht aus den rife_frames/), weil er fuer jeden Sub-Sample
        # RIFE intern aufruft. rife_frames/ wird bei aktiviertem Blur
        # also NICHT als Zwischenprodukt benoetigt -- das spart Disk-
        # Space (keine doppelte Frame-Sequenz auf Disk). 
        # Im motion_grade-Modus entspricht jeder Output-Frame exakt einem Quell-Zeitpunkt 
        # — die RIFE-Aufrufe finden ausschliesslich innerhalb des Blur-Fensters statt, 
        # nicht auf einer separaten zeitgestreckten Achse.
        from src.motion_blur import SubFrameBlurCompositor

        blur_frames_dir = interim_dir / "blur_frames"

        # Cache-Kompatibilitätsprüfung VOR dem Render-Start.
        # Fingerprint beschreibt exakt die Parameter, die den Frame-Inhalt bestimmen.
        current_fp = _render_fingerprint(args, n_output_frames=len(timing_df))
        _check_cache_compatibility(blur_frames_dir, current_fp, force=args.force)

        compositor = SubFrameBlurCompositor(
            source_fps=args.source_fps,
            output_fps=args.output_fps,
            shutter_angle=args.shutter_angle,
            max_subsamples=args.max_subsamples,
            mode=args.mode,
        )
        compositor.process_timing_table(
            timing_df,
            interpolator,
            str(source_frames_dir),
            str(blur_frames_dir),
            max_source_frame_idx=max_source_frame_idx,
        )
        final_frames_dir = blur_frames_dir
        _save_fingerprint(blur_frames_dir, current_fp)
        print(f"[OK] Sub-Frame-Blur-Compositing abgeschlossen: {blur_frames_dir}")

    # --- Schritt 6: FFmpeg-Reassembly ---
    # Im motion_grade-Modus: Container-FPS = source_fps (100fps).
    # Die Clip-Länge in Sekunden ist identisch zum Original.
    # Im speedramp-Modus: Container-FPS = output_fps (wie bisher).
    import subprocess

    if args.mode == "motion_grade":
        reassembly_fps = args.source_fps
    else:
        reassembly_fps = args.output_fps_reassembly or args.output_fps

    output_clip_path = project_root / "data" / "output" / f"{args.clip}_{args.preset}_{args.mode}.mov"
    output_clip_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(reassembly_fps),
        "-i", str(final_frames_dir / "frame_%06d.png"),
        "-c:v", "prores_ks",       # ProRes 4444 fuer verlustfreien Zwischenschritt
        "-profile:v", "4",         # 4444 Profil (Alpha-Kanal-faehig, maximale Qualitaet)
        "-pix_fmt", "yuva444p10le",
        "-vendor", "apl0",
        "-r", str(reassembly_fps),
        str(output_clip_path),
    ]

    print(f"[INFO] Starte FFmpeg-Reassembly: {' '.join(ffmpeg_cmd)}")
    result_proc = subprocess.run(ffmpeg_cmd, capture_output=False)
    if result_proc.returncode != 0:
        raise SystemExit(
            f"FFmpeg-Reassembly fehlgeschlagen (Exit-Code {result_proc.returncode}). "
            f"Sieh dir die FFmpeg-Ausgabe oben an. Die gerenderten Frames bleiben "
            f"erhalten unter: {final_frames_dir}"
        )

    print(f"[OK] Finaler Clip gespeichert: {output_clip_path}")
    print(
        f"\nPipeline vollständig abgeschlossen.\n"
        f"  Quell-Frames:       {source_frames_dir}\n"
        f"  Blur-Frames:        {final_frames_dir}\n"
        f"  Timing-Tabelle:     {timing_csv_path}\n"
        f"  Verifikations-Plot: {interim_dir / 'verification_plot.png'}\n"
        f"  Finaler Output:     {output_clip_path}\n"
    )


if __name__ == "__main__":
    main()
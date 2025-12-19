# Soubor: yt_en2cs_pipeline.py
# Autor: Miloš Pike (VOLTPAJK) + ChatGPT ladění
# Rok: 2025
# Licence: MIT
#
# Funkce:
# - URL -> export\<název_videa>.wav + <název_videa>.en.srt + <název_videa>.cs.srt
# - CLI:
#     python app\yt_en2cs_pipeline.py "<URL>" --audio-only --base --cpu -o export
# - GUI:
#     python app\yt_en2cs_pipeline.py --gui
#
# Download:
# - yt-dlp bez explicitního -f (obchází „Requested format is not available“)
# - v tmp složce vybere nejlepší media soubor (preferuje .webm/.mp4/.m4a/…)
# - ffmpeg:
#     - pro audio-only → WAV 16 kHz mono (ideální pro Whisper)
#     - pro video → MP4 (H.264 + AAC)


import re
import sys
import tempfile
import datetime
import subprocess
import threading
from pathlib import Path

import srt
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel
from tqdm import tqdm

import argostranslate.package as argos_pkg
import argostranslate.translate as argos_tr


# ---------- Defaulty ----------
EXPORT_DIR   = Path("export")
FROM_LANG    = "en"
TO_LANG      = "cs"

# ---------- Logger (sdílený pro CLI i GUI) ----------
LOG_FN = None  # type: ignore[assignment]


def set_logger(fn):
    """Nastaví funkci pro logování (např. append z GUI)."""
    global LOG_FN
    LOG_FN = fn


def log(msg: str):
    """Obecný logger – v CLI tiskne na stdout, v GUI píše do Text widgetu."""
    if LOG_FN is not None:
        try:
            LOG_FN(msg)
        except Exception:
            # kdyby něco selhalo v GUI, fallback na print
            print(msg)
    else:
        print(msg)


# ---------- Pomocné ----------
def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    return name.strip().rstrip(".")


def ensure_argos_en_cs():
    installed = argos_pkg.get_installed_packages()
    if any(p.from_code == FROM_LANG and p.to_code == TO_LANG for p in installed):
        return
    log("[Argos] Stahuji a instaluji model EN->CS ...")
    argos_pkg.update_package_index()
    available = argos_pkg.get_available_packages()
    cand = [p for p in available if p.from_code == FROM_LANG and p.to_code == TO_LANG]
    if not cand:
        raise RuntimeError("Nenalezen balíček EN->CS v indexu Argos.")
    argos_pkg.install_from_path(cand[0].download())
    log("[Argos] Model nainstalován.")


# ---------- ffmpeg helpery ----------
def run_ffmpeg(args):
    """Spustí ffmpeg s danými argumenty, hodí RuntimeError při chybě."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error"] + args
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg selhal:\n{proc.stderr.strip()}")


def convert_to_mp4(src: Path, dst: Path) -> Path:
    """Překóduje libovolné video do MP4 (H.264 + AAC)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    log(f"[FFmpeg] Převádím na MP4: {dst.name}")
    run_ffmpeg([
        "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "128k",
        str(dst),
    ])
    return dst


def convert_to_audio(src: Path, dst: Path) -> Path:
    """Vytáhne audio stopu a překonvertuje ji do WAV (16 kHz mono) – ideál pro Whisper."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    log(f"[FFmpeg] Vytahuji audio do WAV: {dst.name}")
    run_ffmpeg([
        "-i", str(src),
        "-vn",                # bez videa
        "-ac", "1",           # mono
        "-ar", "16000",       # 16 kHz
        "-c:a", "pcm_s16le",  # nekomprimované WAV
        str(dst),
    ])
    return dst


PREFERRED_EXTS = [".webm", ".mp4", ".m4a", ".mkv", ".mp3", ".opus", ".ogg", ".wav"]


def pick_best_media_file(tmp_dir: Path) -> Path:
    """
    Z dočasné složky vybere nejlepší mediální soubor:
    - ignoruje .mhtml, .html, obrázky, json, apod.
    - preferuje přípony v PREFERRED_EXTS
    - pro stejnou příponu bere největší soubor (nejvyšší bitrate)
    """
    candidates: list[Path] = []
    for p in tmp_dir.iterdir():
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in (".mhtml", ".html", ".htm", ".txt", ".json", ".js",
                   ".jpg", ".jpeg", ".png", ".webp", ".gif"):
            continue
        candidates.append(p)

    if not candidates:
        raise RuntimeError("YT: nepodařilo se najít žádný mediální soubor po stažení.")

    def pref_index(ext: str) -> int:
        ext = ext.lower()
        return PREFERRED_EXTS.index(ext) if ext in PREFERRED_EXTS else len(PREFERRED_EXTS)

    # seřadíme: nejdřív podle preference přípony, pak podle velikosti (větší = lepší)
    candidates.sort(key=lambda p: (pref_index(p.suffix), -p.stat().st_size))
    best = candidates[0]
    log(f"[YT] Vybraný stažený soubor: {best.name} (ext={best.suffix}, size={best.stat().st_size} B)")
    return best


def download_youtube(url: str, export_dir: Path, want_video: bool = True) -> Path:
    """
    Stáhne YouTube obsah do dočasné složky pomocí yt-dlp bez explicitního 'format'
    (tím obcházíme „Requested format is not available“),
    vybere nejlepší mediální soubor a zpracuje ho ffmpegem:

      - pokud want_video=True  → <export_dir>/<název>.mp4 (H.264 + AAC)
      - pokud want_video=False → <export_dir>/<název>.wav (16 kHz mono pro Whisper)

    Vrací cestu k výslednému souboru.
    """
    export_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        outtmpl = str(tmp_dir / "video.%(ext)s")

        ydl_opts = {
            "outtmpl": outtmpl,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "config_locations": [],  # ignoruj globální yt-dlp config
        }

        with YoutubeDL(ydl_opts) as ydl:
            log("[YT-dlp] Stahuji přes yt-dlp (bez explicitního formátu)...")
            info = ydl.extract_info(url, download=True)

        # název videa pro pojmenování výstupů
        title = info.get("title") or "video"
        base = sanitize_filename(title)

        # najdeme nejlepší soubor v tmp_dir
        src = pick_best_media_file(tmp_dir)

        if want_video:
            dst = export_dir / f"{base}.mp4"
            log(f"[YT] Konvertuji video do MP4: {dst.name}")
            return convert_to_mp4(src, dst)
        else:
            dst = export_dir / f"{base}.wav"
            log(f"[YT] Vytahuji audio pro Whisper: {dst.name}")
            return convert_to_audio(src, dst)


# ---------- Whisper + Argos ----------
def transcribe_to_srt(
    media_path: Path,
    srt_out: Path,
    model_name: str = "small.en",
    device: str = "auto",
    compute_type: str = "auto",
    duration_sec: float | None = None
):
    """Přepis do EN SRT."""
    log(f"[FW] Načítám model: {model_name} (device={device}, compute_type={compute_type})")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    log(f"[FW] Přepisuji: {media_path}")
    # v GUI nechceme tqdm progress bar – bude se zobrazovat jen v konzoli
    use_tqdm = (LOG_FN is None)
    pbar = None
    last = 0.0
    if duration_sec and duration_sec > 0 and use_tqdm:
        pbar = tqdm(total=duration_sec, unit="s", desc="Přepis", ncols=80)

    segments, info = model.transcribe(
        str(media_path),
        language="en",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=5
    )

    subs = []
    next_log_t = 60.0  # v GUI pošleme info každou cca minutu audio času
    for i, seg in enumerate(segments, 1):
        subs.append(
            srt.Subtitle(
                index=i,
                start=datetime.timedelta(seconds=seg.start),
                end=datetime.timedelta(seconds=seg.end),
                content=seg.text.strip()
            )
        )
        if pbar:
            inc = max(0.0, float(seg.end) - last)
            pbar.update(inc)
            last = float(seg.end)

        # jednoduchý heartbeat pro GUI – ať je vidět, že to žije
        if LOG_FN is not None:
            if float(seg.end) >= next_log_t:
                log(f"[FW] Přepsáno minimálně {int(seg.end)} s audio...")
                next_log_t += 60.0

    if pbar:
        pbar.n = pbar.total
        pbar.refresh()
        pbar.close()

    srt_out.parent.mkdir(parents=True, exist_ok=True)
    with open(srt_out, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))
    log(f"[FW] Uloženo: {srt_out}")


def translate_srt_en2cs(srt_in: Path, srt_out: Path):
    log("[Argos] Kontroluji EN->CS model...")
    ensure_argos_en_cs()
    with open(srt_in, "r", encoding="utf-8", errors="ignore") as f:
        en_text = f.read()
    subs = list(srt.parse(en_text))

    log("[Argos] Překládám titulky EN→CS...")
    out_subs = []
    for sub in subs:
        cz = argos_tr.translate(sub.content, FROM_LANG, TO_LANG).strip()
        out_subs.append(
            srt.Subtitle(index=sub.index, start=sub.start, end=sub.end, content=cz)
        )

    srt_out.parent.mkdir(parents=True, exist_ok=True)
    with open(srt_out, "w", encoding="utf-8") as f:
        f.write(srt.compose(out_subs))
    log(f"[Argos] Uloženo: {srt_out}")


# ---------- CLI pipeline ----------
def run_pipeline(url: str, export_dir: Path, model_name: str,
                 device: str, compute_type: str, want_video: bool):
    duration = None  # bez metadat – progress bar bude jen podle segmentů
    log(f"[YT] Stahuji {'video' if want_video else 'audio'} z URL: {url}")
    media_path = download_youtube(url, export_dir, want_video=want_video)
    base = media_path.stem

    srt_en = export_dir / f"{base}.en.srt"
    srt_cs = export_dir / f"{base}.cs.srt"

    transcribe_to_srt(media_path, srt_en, model_name, device, compute_type, duration)
    translate_srt_en2cs(srt_en, srt_cs)

    log("\n[OK] Hotovo.")
    if media_path.suffix.lower() == ".mp4":
        log(f"     Video:      {media_path}")
    log(f"     EN titulky: {srt_en}")
    log(f"     CZ titulky: {srt_cs}")


def parse_args(argv):
    # jednoduchý parser s podporou GUI
    args = {
        "url": None,
        "export": EXPORT_DIR,
        "model": "small.en",
        "device": "auto",
        "compute": "auto",
        "video": True,
        "gui": False,
    }
    flags = set(argv[1:])

    # GUI mód?
    if "--gui" in flags or "-g" in flags:
        args["gui"] = True
        return args

    # URL (první ne-flag argument)
    nonflags = [a for a in argv[1:] if not a.startswith("-")]
    if nonflags:
        args["url"] = nonflags[0]

    # model
    if "--tiny" in flags:   args["model"] = "tiny.en"
    if "--base" in flags:   args["model"] = "base.en"
    if "--small" in flags:  args["model"] = "small.en"
    if "--medium" in flags: args["model"] = "medium"
    if "--large" in flags:  args["model"] = "large-v2"

    # device
    if "--cpu" in flags:    args["device"] = "cpu"
    if "--cuda" in flags:   args["device"] = "cuda"
    if "--auto" in flags:   args["device"] = "auto"

    # compute type
    if "--int8" in flags:           args["compute"] = "int8"
    if "--int8-fp16" in flags:      args["compute"] = "int8_float16"
    if "--fp16" in flags or "--f16" in flags: args["compute"] = "float16"
    if "--fp32" in flags or "--f32" in flags: args["compute"] = "float32"

    # jen audio?
    if "--audio-only" in flags: args["video"] = False

    # export dir
    for i, a in enumerate(argv):
        if a in ("--export", "-o") and i + 1 < len(argv):
            args["export"] = Path(argv[i + 1])

    return args


# ---------- GUI ----------
def autodetect_defaults():
    model = "small.en"
    device = "auto"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            # když je CUDA, bývá OK i small/medium; necháme small.en
    except Exception:
        pass
    return model, device


def run_gui():
    import tkinter as tk
    from tkinter import ttk, messagebox

    root = tk.Tk()
    root.title("YT video → EN/CZ titulky  |   Miloš Pike (VOLTPAJK)")

    model_default, device_default = autodetect_defaults()

    url_var = tk.StringVar()
    model_var = tk.StringVar(value=model_default)
    device_var = tk.StringVar(value=device_default)
    video_var = tk.BooleanVar(value=True)

    frm = ttk.Frame(root, padding=12)
    frm.grid(sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    ttk.Label(frm, text="YouTube URL:").grid(row=0, column=0, sticky="w")
    url_entry = ttk.Entry(frm, textvariable=url_var, width=70)
    url_entry.grid(row=0, column=1, columnspan=3, sticky="ew", padx=(6, 0))

    # Modely – vlastní rámec
    ttk.Label(frm, text="Model:").grid(row=1, column=0, sticky="w", pady=(8, 0))
    models_frm = ttk.Frame(frm)
    models_frm.grid(row=1, column=1, columnspan=5, sticky="w", pady=(8, 0))

    for i, (label, val) in enumerate([
        ("tiny", "tiny.en"),
        ("base", "base.en"),
        ("small", "small.en"),
        ("medium", "medium"),
        ("large", "large-v2"),
    ]):
        ttk.Radiobutton(models_frm, text=label, value=val, variable=model_var) \
            .grid(row=0, column=i, sticky="w", padx=6)

    ttk.Label(frm, text="Zařízení:").grid(row=2, column=0, sticky="w", pady=(8, 0))
    for i, (label, val) in enumerate([("auto", "auto"), ("CPU", "cpu"), ("CUDA", "cuda")]):
        ttk.Radiobutton(frm, text=label, value=val, variable=device_var) \
            .grid(row=2, column=1 + i, sticky="w", padx=4, pady=(8, 0))

    ttk.Checkbutton(frm, text="Stáhnout celé video (MP4)", variable=video_var) \
        .grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

    log_txt = tk.Text(frm, height=16, width=90)
    log_txt.grid(row=4, column=0, columnspan=5, sticky="nsew", pady=(8, 0))
    frm.rowconfigure(4, weight=1)
    frm.columnconfigure(3, weight=1)

    def append(msg: str):
        log_txt.insert("end", msg + "\n")
        log_txt.see("end")
        root.update_idletasks()

    def run_job():
        url = url_var.get().strip()
        if not url:
            messagebox.showerror("Chyba", "Zadej YouTube URL.")
            return
        try:
            # nasměrujeme globální logger do GUI
            set_logger(append)
            append("[Start] Zpracovávám…")
            run_pipeline(
                url=url,
                export_dir=EXPORT_DIR,
                model_name=model_var.get(),
                device=device_var.get(),
                compute_type="auto",
                want_video=video_var.get(),
            )
            append("[OK] Hotovo. Výstup v ./export")
        except Exception as e:
            append(f"[Error] {e}")
            messagebox.showerror("Chyba", str(e))
        finally:
            # po doběhnutí můžeme logger nechat, ať další běh jde taky do GUI
            pass

    def on_start():
        threading.Thread(target=run_job, daemon=True).start()

    btn = ttk.Button(frm, text="Start", command=on_start)
    btn.grid(row=5, column=0, sticky="w", pady=(8, 0))

    root.mainloop()


# ---------- main ----------
if __name__ == "__main__":
    args = parse_args(sys.argv)

    # GUI režim
    if args.get("gui"):
        run_gui()
        sys.exit(0)

    # CLI režim – logger necháme jako print
    if not args["url"]:
        print("Použití:\n"
              "  python app\\yt_en2cs_pipeline_funkcni.py <YouTube_URL> [volby]\n"
              "Volby modelu: --tiny | --base | --small | --medium | --large\n"
              "Zařízení:     --cpu | --cuda | --auto (výchozí)\n"
              "Výstupní složka: --export <cesta> nebo -o <cesta>\n"
              "Jen audio (rychlejší, menší): --audio-only\n"
              "GUI mód: --gui nebo -g\n"
              "Příklad:\n"
              "  python app\\yt_en2cs_pipeline_funkcni.py "
              "\"https://youtu.be/...\" --base --cpu -o export")
        sys.exit(1)

    # CLI běh
    run_pipeline(
        url=args["url"],
        export_dir=args["export"],
        model_name=args["model"],
        device=args["device"],
        compute_type=args["compute"],
        want_video=args["video"],
    )

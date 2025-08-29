# Soubor: yt_en2cs_pipeline.py
# Autor: Miloš Pike (VOLTPAJK)
# Rok: 2025
# Licence: MIT
#
# Popis:
# - URL -> export\<název>.mp4 + <název>.en.srt + <název>.cs.srt
# - CLI: model volíš přepínači --tiny/--base/--small/--medium/--large, zařízení --device auto|cpu|cuda
#        progress bar přes tqdm, volba GUI: --gui
# - GUI: jednoduché Tkinter okno (URL, model, device) + autodetekce rozumných defaultů


import os, re, sys, tempfile, datetime, threading, shutil
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

# ---------- Pomocné ----------
def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    return name.strip().rstrip(".")

def ensure_argos_en_cs():
    installed = argos_pkg.get_installed_packages()
    if any(p.from_code == FROM_LANG and p.to_code == TO_LANG for p in installed):
        return
    print("[Argos] Stahuji a instaluji model EN->CS ...")
    argos_pkg.update_package_index()
    available = argos_pkg.get_available_packages()
    cand = [p for p in available if p.from_code == FROM_LANG and p.to_code == TO_LANG]
    if not cand:
        raise RuntimeError("Nenalezen balíček EN->CS v indexu Argos.")
    argos_pkg.install_from_path(cand[0].download())
    print("[Argos] Model nainstalován.")

def probe_title_and_duration(url: str):
    """Získá title a duration bez stahování obsahu."""
    opts = {"quiet": True, "no_warnings": True}
    with YoutubeDL(opts) as y:
        info = y.extract_info(url, download=False)
    title = info.get("title") or "video"
    duration = info.get("duration")
    return title, duration

def download_youtube(url: str,
                     export_dir: Path,
                     want_video: bool = True,
                     on_progress=None,
                     suppress_console_progress: bool = False) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    title, _ = probe_title_and_duration(url)
    base  = sanitize_filename(title)

    def _hook(d):
        if on_progress is None:
            return
        try:
            if d.get("status") == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                done  = d.get("downloaded_bytes") or 0
                if total:
                    on_progress(min(100.0, 100.0 * done / total))
            elif d.get("status") == "finished":
                on_progress(100.0)
        except Exception:
            pass

    common_opts = {
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_hook] if on_progress else [],
    }
    if suppress_console_progress:
        common_opts["noprogress"] = True

    if want_video:
        outtmpl = str(export_dir / f"{base}.%(ext)s")
        ydl_opts = {
            **common_opts,
            "format": "bestvideo+bestaudio/best",
            "merge_output_format": "mp4",
            "outtmpl": outtmpl,
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # preferovaná cesta:
            merged = export_dir / f"{base}.mp4"
            if merged.exists():
                return merged
            # fallback
            pf = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
            if pf.exists():
                return pf
            cand = list(export_dir.glob(f"{base}*.mp4"))
            if cand:
                return cand[0]
            raise RuntimeError("Nepodařilo se najít stažené MP4.")
    else:
        with tempfile.TemporaryDirectory() as td:
            tmp_dir = Path(td)
            outtmpl = str(tmp_dir / f"{base}.%(ext)s")
            ydl_opts = {
                **common_opts,
                "format": "bestaudio/best",
                "outtmpl": outtmpl,
            }
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                # najdi audio soubor
                for p in tmp_dir.iterdir():
                    if p.suffix.lower() in (".m4a", ".webm", ".opus", ".mp3"):
                        # přesun do exportu kvůli jednotnému umístění
                        target = export_dir / p.name
                        shutil.move(str(p), target)
                        return target
            raise RuntimeError("Audio se nestáhlo.")

def transcribe_to_srt(media_path: Path, srt_out: Path,
                      model_name: str = "small.en",
                      device: str = "auto",
                      compute_type: str = "auto",
                      duration_sec: float | None = None,
                      on_progress=None):
    print(f"[FW] Načítám model: {model_name} (device={device}, compute_type={compute_type})")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    print(f"[FW] Přepisuji: {media_path}")
    use_gui_progress = callable(on_progress) and duration_sec and duration_sec > 0

    pbar = None
    last = 0.0
    if use_gui_progress:
        on_progress(0.0)
    elif duration_sec and duration_sec > 0:
        pbar = tqdm(total=duration_sec, unit="s", desc="Přepis", ncols=80)

    segments, info = model.transcribe(
        str(media_path),
        language="en",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=5
    )

    subs = []
    for i, seg in enumerate(segments, 1):
        subs.append(
            srt.Subtitle(
                index=i,
                start=datetime.timedelta(seconds=seg.start),
                end=datetime.timedelta(seconds=seg.end),
                content=seg.text.strip()
            )
        )
        if use_gui_progress and duration_sec:
            last = float(seg.end)
            on_progress(min(100.0, 100.0 * last / float(duration_sec)))
        elif pbar:
            inc = max(0.0, float(seg.end) - last)
            pbar.update(inc)
            last = float(seg.end)

    if pbar:
        pbar.n = pbar.total
        pbar.refresh()
        pbar.close()

    srt_out.parent.mkdir(parents=True, exist_ok=True)
    with open(srt_out, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))
    print(f"[FW] Uloženo: {srt_out}")

def translate_srt_en2cs(srt_in: Path, srt_out: Path):
    ensure_argos_en_cs()
    with open(srt_in, "r", encoding="utf-8", errors="ignore") as f:
        en_text = f.read()
    subs = list(srt.parse(en_text))

    out_subs = []
    for sub in subs:
        cz = argos_tr.translate(sub.content, FROM_LANG, TO_LANG).strip()
        out_subs.append(
            srt.Subtitle(index=sub.index, start=sub.start, end=sub.end, content=cz)
        )

    srt_out.parent.mkdir(parents=True, exist_ok=True)
    with open(srt_out, "w", encoding="utf-8") as f:
        f.write(srt.compose(out_subs))
    print(f"[Argos] Uloženo: {srt_out}")

# ---------- CLI ----------
def run_pipeline(url: str, export_dir: Path, model_name: str,
                 device: str, compute_type: str, want_video: bool,
                 on_dl_progress=None, on_tr_progress=None,
                 suppress_console_progress=False):
    title, duration = probe_title_and_duration(url)
    safe = sanitize_filename(title)

    # download (video.mp4 nebo audio.*)
    print(f"[YT] Stahuji {'video' if want_video else 'audio'} z URL: {url}")
    media_path = download_youtube(
        url, export_dir, want_video=want_video,
        on_progress=on_dl_progress,
        suppress_console_progress=suppress_console_progress
    )
    base = media_path.stem  # název bez přípony

    # sjednocení názvu SRT vedle média:
    srt_en = export_dir / f"{base}.en.srt"
    srt_cs = export_dir / f"{base}.cs.srt"

    # transcribe
    transcribe_to_srt(
        media_path, srt_en, model_name, device, compute_type, duration,
        on_progress=on_tr_progress
    )

    # translate
    translate_srt_en2cs(srt_en, srt_cs)

    print("\n[OK] Hotovo.")
    if media_path.suffix.lower() == ".mp4":
        print(f"     Video:      {media_path}")
    print(f"     EN titulky: {srt_en}")
    print(f"     CZ titulky: {srt_cs}")

def parse_args(argv):
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

    # URL 
    nonflags = [a for a in argv[1:] if not a.startswith("-")]
    if nonflags:
        args["url"] = nonflags[0]

    # model přepínače 
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
    import threading, sys

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
    url_entry.grid(row=0, column=1, columnspan=5, sticky="ew", padx=(6,0))
    frm.grid_columnconfigure(5, weight=1)

    # Modely – vlastní rámec
    ttk.Label(frm, text="Model:").grid(row=1, column=0, sticky="w", pady=(8,0))
    models_frm = ttk.Frame(frm)
    models_frm.grid(row=1, column=1, columnspan=5, sticky="w", pady=(8,0))

    for i, (label, val) in enumerate([
        ("tiny", "tiny.en"),
        ("base", "base.en"),
        ("small", "small.en"),
        ("medium", "medium"),
        ("large", "large-v2"),
    ]):
        ttk.Radiobutton(models_frm, text=label, value=val, variable=model_var)\
            .grid(row=0, column=i, sticky="w", padx=6)

    # Zařízení
    ttk.Label(frm, text="Zařízení:").grid(row=2, column=0, sticky="w", pady=(8,0))
    for i, (label, val) in enumerate([("auto","auto"), ("CPU","cpu"), ("CUDA","cuda")]):
        ttk.Radiobutton(frm, text=label, value=val, variable=device_var)\
            .grid(row=2, column=1+i, sticky="w", padx=4, pady=(8,0))

    ttk.Checkbutton(frm, text="Stáhnout celé video (MP4)", variable=video_var)\
        .grid(row=3, column=0, columnspan=2, sticky="w", pady=(8,0))

    # Progress bary
    ttk.Label(frm, text="Stahování:").grid(row=4, column=0, sticky="w", pady=(8,0))
    dl_pb = ttk.Progressbar(frm, mode="determinate", length=400, maximum=100)
    dl_pb.grid(row=4, column=1, columnspan=5, sticky="w", pady=(8,0))

    ttk.Label(frm, text="Přepis:").grid(row=5, column=0, sticky="w", pady=(4,0))
    tr_pb = ttk.Progressbar(frm, mode="determinate", length=400, maximum=100)
    tr_pb.grid(row=5, column=1, columnspan=5, sticky="w", pady=(4,0))

    # Log okno
    log = tk.Text(frm, height=12, width=90)
    log.grid(row=6, column=0, columnspan=6, sticky="nsew", pady=(8,0))
    frm.rowconfigure(6, weight=1)
    frm.columnconfigure(3, weight=1)

    def log_line(msg: str):
        msg = msg.strip()
        if not msg:
            return
        log.insert("end", msg + "\n")
        log.see("end")

    # GUI sink – filtr
    class CleanGuiSink:
        noisy_prefixes = (
            "[download]", "Přepis:",
            "vocabulary.txt", "tokenizer.json", "config.json", "model.bin",
        )
        def write(self, s: str):
            s = s.replace("\r", "")
            if "FutureWarning" in s and "stanza" in s:
                return
            if any(s.lstrip().startswith(p) for p in self.noisy_prefixes):
                return
            if s.strip():
                log.after(0, log_line, s)
        def flush(self): pass

    # konzole + GUI
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, s):
            for st in self.streams:
                try:
                    st.write(s)
                except Exception:
                    pass
        def flush(self):
            for st in self.streams:
                try:
                    st.flush()
                except Exception:
                    pass

    # Zapoj duální výstup: původní konzole + čistý GUI sink
    sys_stdout_orig, sys_stderr_orig = sys.stdout, sys.stderr
    gui_sink = CleanGuiSink()
    sys.stdout = Tee(sys_stdout_orig, gui_sink)
    sys.stderr = Tee(sys_stderr_orig, gui_sink)

    # GUI callbacky pro progress 
    def on_dl_progress(pct: float):
        dl_pb["value"] = pct
    def on_tr_progress(pct: float):
        tr_pb["value"] = pct

    def run_job():
        url = url_var.get().strip()
        if not url:
            messagebox.showerror("Chyba", "Zadej YouTube URL.")
            return
        try:
            print("[Start] Zpracovávám…")
            run_pipeline(
                url=url,
                export_dir=EXPORT_DIR,
                model_name=model_var.get(),
                device=device_var.get(),
                compute_type="auto",
                want_video=video_var.get(),
                on_dl_progress=lambda p: root.after(0, on_dl_progress, p),
                on_tr_progress=lambda p: root.after(0, on_tr_progress, p),
                suppress_console_progress=False,
            )
            print("[OK] Hotovo. Výstup v ./export")
            dl_pb["value"] = 100
            tr_pb["value"] = 100
        except Exception as e:
            print(f"[Error] {e}")
            messagebox.showerror("Chyba", str(e))

    def on_start():
        threading.Thread(target=run_job, daemon=True).start()

    btn = ttk.Button(frm, text="Start", command=on_start)
    btn.grid(row=7, column=0, sticky="w", pady=(8,0))

    root.mainloop()

# ---------- main ----------
if __name__ == "__main__":
    args = parse_args(sys.argv)
    if args["gui"]:
        run_gui()
        sys.exit(0)

    if not args["url"]:
        print("Použití:\n  python app\\yt_en2cs_pipeline.py <YouTube_URL> [volby]\n"
              "Volby modelu: --tiny | --base | --small | --medium | --large\n"
              "Zařízení: --cpu | --cuda | --auto (výchozí)\n"
              "Výstupní složka: --export <cesta> nebo -o <cesta>\n"
              "Jen audio (rychlejší, menší): --audio-only\n"
              "GUI mód: --gui nebo -g\n"
              "Příklad:\n  python app\\yt_en2cs_pipeline.py \"https://youtu.be/…\" --base --cpu -o export")
        sys.exit(1)

    # CLI režim – nic se nemění, progressy (tqdm) zůstávají v konzoli
    run_pipeline(
        url=args["url"],
        export_dir=args["export"],
        model_name=args["model"],
        device=args["device"],
        compute_type=args["compute"],
        want_video=args["video"],
        on_dl_progress=None,            # v CLI nepoužíváme GUI progress
        on_tr_progress=None,
        suppress_console_progress=False # v CLI necháme default výstupy
    )

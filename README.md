# YT → EN/CZ titulky (offline) + volitelný download videa

## Co dělá tento nástroj:
1) stáhne YouTube (video nebo jen audio),
2) přepíše mluvené slovo do **anglických titulků .srt** (faster-whisper),
3) **offline** přeloží titulky do **češtiny .srt** (Argos Translate),
4) vše uloží do složky `export/` jako:
   - `<název>.mp4`  (pokud stahuješ celé video)
   - `<název>.en.srt` (anglické titulky)
   - `<název>.cs.srt` (české titulky)

Funguje plně **offline** po prvním stažení modelů (Whisper a Argos).
Vše je **open-source**.

---

## Požadavky
- Windows 10/11, Python 3.11+ (doporučeno 3.12)
- nainstalovaný **FFmpeg** v PATH (máš-li `ffmpeg -version`, je OK)
- LINUX (libovolná distribuce): návod je sice připravený pro Windows, ale zkušenější uživatelé Linuxu podle něj snadno najdou odpovídající postup.  

---

## Instalace (CMD)
Doporučuji izolovat projekt do `venv`.

v cmd  
- vytvoř složku projektu  
`mkdir D:\Projects\subs-translator`  
`cd D:\Projects\subs-translator`  

- virtuální prostředí  
`python -m venv venv`  
`venv\Scripts\activate.bat`  

- závislosti (CPU varianta)  
`pip install --upgrade pip`  
`pip install -r requirements.txt`  


**Volitelně: GPU (NVIDIA CUDA)**  
Chceš-li rychlejší běh na GPU, doinstaluj PyTorch s CUDA koly:  
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`  
Pokud CUDA nemáš, tenhle krok vynech. Skript poběží na CPU.  
*Záměrně není zahrnut v requirements.txt*  



## Struktura projektu
```
D:\Projects\subs-translator\
│  requirements.txt  
│  README.md  
│  
├─ venv\  
└─ app\  
   └─ yt_en2cs_pipeline.py  

```

## Použití – CLI
**Aktivní venv (vždy):**  
`venv\Scripts\activate.bat`

Základní použití (stáhne video MP4, udělá EN i CZ titulky):
python app\yt_en2cs_pipeline.py "https://youtu.be/…"

## Přepínače

- volba modelu (rychlost/přesnost)  
--tiny | --base | --small (výchozí) | --medium | --large

- zařízení  
--auto (výchozí) | --cpu | --cuda

- jen audio (bez videa, rychlejší download)  
--audio-only

- výstupní složka  
--export *cesta* ;nebo -o *cesta*

- GUI mód  **<doporučeno>**  
--gui nebo -g (otevře jednoduché okno s URL, výběrem modelu a zařízení)



## Příklady:
- nejrychlejší přepis (tiny), jen audio:  
`python app\yt_en2cs_pipeline.py "https://youtu.be/…" --tiny --audio-only`  

- rychlejší CPU přepis (base), stáhnout celé video:  
`python app\yt_en2cs_pipeline.py "https://youtu.be/…" --base --cpu`  

- GPU (pokud máš CUDA a Torch CUDA nainstalován):  
`python app\yt_en2cs_pipeline.py "https://youtu.be/…" --small --cuda`  

- GUI režim:  
`python app\yt_en2cs_pipeline.py --gui`  

---

## Co se děje uvnitř

yt-dlp stáhne video (MP4) nebo audio (M4A/WEBM).  
faster-whisper (model např. small.en) přepíše zvuk → *.en.srt.  
FFmpeg se postará o extrakci a převod audia (16 kHz mono) automaticky.  
Argos Translate offline přeloží *.en.srt → *.cs.srt (první běh stáhne EN→CS balíček).  

## Tipy

VLC načte titulky automaticky, když se jmenují stejně jako video (což skript dělá: *název*.en.srt, *název*.cs.srt).
Pokud chceš jen CZ, prostě ignoruj *.en.srt.  
První běh může být delší (stažení modelů). Další běhy jsou už offline a rychlé.  
Varování typu pkg_resources is deprecated a symlink warning z huggingface_hub můžeš ignorovat.  
Chceš-li potlačit symlink warning na Windows: zapni Developer Mode nebo nastav proměnnou HF_HUB_DISABLE_SYMLINKS_WARNING=1.  

## Troubleshooting

- „ffmpeg není rozpoznán“ → doplň FFmpeg do PATH nebo použij plnou cestu.  
- „CUDA not available“ → buď nemáš NVIDIA GPU, nebo chybí CUDA build PyTorch (viz krok GPU).  
- „Rate limit / 429“ na YouTube → zopakuj později; případně použij --audio-only (méně požadavků).  
- „Diakritika v SRT“ → soubory jsou UTF-8; VLC to umí. Pokud ne, zkontroluj kódování souboru (Notepad: Uložit jako → UTF-8).  

---
# Licencování a přiznání zdrojů

## Moje licence
Tento projekt je licencován pod [licencí MIT](./LICENSE).  
To znamená, že kód je možné volně používat, kopírovat, upravovat a distribuovat, včetně komerčního využití, za podmínky zachování této licence a uvedení autorství.

Copyright (c) 2025 [Milos PIke]

---

## Použité závislosti a jejich licence

- [yt-dlp](https://pypi.org/project/yt-dlp/)  
  Licence: [Unlicense](https://unlicense.org/)  
  Zdrojový kód: https://github.com/yt-dlp/yt-dlp

- [faster-whisper](https://pypi.org/project/faster-whisper/)  
  Licence: [MIT](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE)  
  Zdrojový kód: https://github.com/SYSTRAN/faster-whisper

- [whisper-ctranslate2](https://pypi.org/project/whisper-ctranslate2/)  
  Licence: [MIT](https://github.com/jordimas/whisper-ctranslate2/blob/master/LICENSE)  
  Autor: Jordi Mas  
  Zdrojový kód: https://github.com/jordimas/whisper-ctranslate2

- [Argos Translate](https://pypi.org/project/argostranslate/)  
  Licence: [MIT / CC0](https://github.com/argosopentech/argos-translate/blob/master/LICENSE)  
  Autor: Argos Open Technologies, LLC  
  Zdrojový kód: https://github.com/argosopentech/argos-translate



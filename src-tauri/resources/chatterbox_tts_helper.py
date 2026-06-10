#!/usr/bin/env python3
"""Helper TTS Chatterbox Multilingual (Resemble AI) — voice cloning zero-shot.

Stesso protocollo stdio degli helper Kokoro/Piper (avviato da tts.rs):
  stdin:  {"text": "...", "voice": "<nome-voce-clonata>", "lang": "it", "speed": 1.0}
          {"cmd": "quit"}
  stdout: [u32 LE count][u32 LE sample_rate][count * f32 LE]
          count == 0  -> errore (dettagli su stderr)
  stderr: log ed errori

Cloning: `voice` e' il nome (senza estensione) di un wav di riferimento in
$KOKORO_MODEL_DIR/cloned_voices/ (creato da POST /voices/clone). Se la voce
non esiste si usa la voce builtin del modello. I conditionals del riferimento
vengono calcolati una volta e tenuti in cache (cambiare voce costa solo la
prima sintesi).

`lang` e' il language_id di Chatterbox Multilingual (it, en, fr, de, es, pt...).
`speed` non e' supportato dal modello: viene ignorato (avviso su stderr).

Ambiente: il venv dedicato (chatterbox-venv, creato con --system-site-packages
per riusare il torch CUDA di sistema). Dipendenza: pip install chatterbox-tts.
Alla prima esecuzione scarica i pesi da HuggingFace (~3 GB, una tantum).
"""
from __future__ import annotations

import json
import os
import struct
import sys


def log(*a) -> None:
    print("[chatterbox_tts_helper]", *a, file=sys.stderr, flush=True)


def write_error(stdout) -> None:
    stdout.write(struct.pack("<II", 0, 0))
    stdout.flush()


def cloned_dir() -> str:
    base = os.environ.get("KOKORO_MODEL_DIR") or "."
    d = os.path.join(base, "cloned_voices")
    os.makedirs(d, exist_ok=True)
    return d


def main() -> int:
    try:
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except Exception as e:  # noqa: BLE001
        log(f"import chatterbox fallito (venv chatterbox-venv? pip install "
            f"chatterbox-tts?): {e}")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"caricamento Chatterbox Multilingual su {device} "
        "(primo avvio: scarica i pesi da HuggingFace)...")
    try:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    except Exception as e:  # noqa: BLE001
        log(f"caricamento modello fallito: {e}")
        return 1
    sample_rate = int(model.sr)
    log(f"pronto (sr={sample_rate})")

    stdout = sys.stdout.buffer
    # Cache dei conditionals per voce: nome -> True (gia' preparati nel modello).
    # Il modello tiene UN set di conditionals alla volta: si rigenerano solo
    # quando cambia la voce richiesta.
    current_voice: str | None = object()  # sentinella != None (None = builtin)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            log(f"richiesta JSON non valida: {line!r}")
            write_error(stdout)
            continue
        if req.get("cmd") == "quit":
            log("ricevuto quit, esco")
            break

        text = (req.get("text") or "").strip()
        if not text:
            write_error(stdout)
            continue
        lang = (req.get("lang") or "it").split("-")[0].lower()
        voice = (req.get("voice") or "").strip() or None
        if req.get("speed") not in (None, 1.0):
            log("nota: 'speed' non supportato da Chatterbox, ignorato")

        try:
            # Voce clonata: prepara i conditionals solo se la voce e' cambiata.
            if voice != current_voice:
                ref = None
                if voice:
                    for ext in (".wav", ".mp3", ".flac", ".ogg"):
                        cand = os.path.join(cloned_dir(), voice + ext)
                        if os.path.isfile(cand):
                            ref = cand
                            break
                if voice and ref is None:
                    log(f"voce clonata '{voice}' non trovata in {cloned_dir()}: "
                        "uso la voce builtin")
                if ref:
                    model.prepare_conditionals(ref)
                    log(f"voce clonata attiva: {os.path.basename(ref)}")
                current_voice = voice if ref else None

            with torch.inference_mode():
                wav = model.generate(text, language_id=lang)
            samples = wav.squeeze().detach().cpu().numpy().astype("<f4")
            stdout.write(struct.pack("<II", samples.size, sample_rate))
            stdout.write(samples.tobytes())
            stdout.flush()
        except Exception as e:  # noqa: BLE001 - segnala e continua
            log(f"sintesi fallita per {text[:60]!r}: {e}")
            write_error(stdout)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

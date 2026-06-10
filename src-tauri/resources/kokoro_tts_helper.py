"""Helper TTS Kokoro per Handy — processo persistente.

Handy lo avvia una volta per sessione di chiamata e lo tiene vivo, cosi' il
modello Kokoro si carica una sola volta (non a ogni frase).

Protocollo (parla con `src-tauri/src/tts.rs`):
  stdin  : una riga JSON per richiesta
             {"text": "...", "voice": "if_sara", "lang": "it", "speed": 1.0}
           riga speciale di chiusura:
             {"cmd": "quit"}
  stdout : per ogni richiesta, binario little-endian:
             [u32 count][u32 sample_rate][count * f32]
           count == 0  -> errore (dettagli su stderr)
  stderr : log ed errori

Requisiti:
  pip install kokoro-onnx
Modelli (kokoro-v1.0.onnx, voices-v1.0.bin): cercati in
  - $KOKORO_MODEL_DIR
  - ./models accanto a questo script, ../models, .
"""

from __future__ import annotations

import glob
import json
import os
import struct
import sys


def log(msg: str) -> None:
    print(f"[kokoro_tts_helper] {msg}", file=sys.stderr, flush=True)


def find_models() -> tuple[str, str]:
    """Individua un modello kokoro*.onnx e voices-v1.0.bin.

    Accetta qualsiasi variante (kokoro-v1.0.onnx, .fp16.onnx, .int8.onnx):
    se ne sono presenti piu' di una, sceglie la prima in ordine alfabetico.
    """
    candidates: list[str] = []
    env_dir = os.environ.get("KOKORO_MODEL_DIR")
    if env_dir:
        candidates.append(env_dir)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates += [
        os.path.join(here, "models"),
        os.path.join(here, "..", "models"),
        here,
    ]
    # Variante specifica scelta dall'utente (Settings -> Models -> TTS).
    forced = os.environ.get("HANDY_TTS_MODEL_FILE")
    for d in candidates:
        voices = os.path.join(d, "voices-v1.0.bin")
        if not os.path.isfile(voices):
            continue
        # Se l'utente ha scelto una variante e c'e', usala.
        if forced:
            forced_path = os.path.join(d, forced)
            if os.path.isfile(forced_path):
                return forced_path, voices
        onnx_matches = sorted(glob.glob(os.path.join(d, "kokoro*.onnx")))
        if onnx_matches:
            return onnx_matches[0], voices
    raise FileNotFoundError(
        "kokoro*.onnx / voices-v1.0.bin non trovati. Esegui setup/setup_kokoro.py "
        "oppure metti i file in $KOKORO_MODEL_DIR o in resources/models. "
        "Cercato in: " + ", ".join(candidates)
    )


def write_error(stdout) -> None:
    """Segnala un errore al chiamante: count=0, rate=0."""
    stdout.write(struct.pack("<II", 0, 0))
    stdout.flush()


def main() -> int:
    try:
        from kokoro_onnx import Kokoro
    except ImportError:
        log("kokoro-onnx non installato. Esegui: pip install kokoro-onnx")
        return 1

    try:
        import numpy as np
    except ImportError:
        log("numpy non installato. Esegui: pip install numpy")
        return 1

    try:
        onnx_path, voices_path = find_models()
    except FileNotFoundError as e:
        log(str(e))
        return 1

    log(f"caricamento modello Kokoro: {onnx_path}")
    kokoro = Kokoro(onnx_path, voices_path)
    log("pronto")

    stdout = sys.stdout.buffer

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

        voice = req.get("voice", "if_sara")
        lang = req.get("lang", "it")
        speed = float(req.get("speed", 1.0))

        try:
            samples, sample_rate = kokoro.create(
                text, voice=voice, speed=speed, lang=lang
            )
            pcm = np.ascontiguousarray(samples, dtype="<f4")
            stdout.write(struct.pack("<II", pcm.size, int(sample_rate)))
            stdout.write(pcm.tobytes())
            stdout.flush()
        except Exception as e:  # noqa: BLE001 - qualunque errore -> segnala e continua
            log(f"sintesi fallita per {text!r}: {e}")
            write_error(stdout)

    return 0


if __name__ == "__main__":
    sys.exit(main())

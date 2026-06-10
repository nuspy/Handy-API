#!/usr/bin/env python3
"""Helper TTS Piper per Handy — stesso protocollo dell'helper Kokoro.

Avviato come subprocess persistente da `tts.rs` (struct KokoroTts, riusata per
qualsiasi engine helper-based). Carica una voce Piper (.onnx + .onnx.json) e
sintetizza su richiesta.

Protocollo (identico a kokoro_tts_helper.py):
  stdin:  {"text": "...", "voice": "...", "lang": "...", "speed": 1.0}
          {"cmd": "quit"}
  stdout: [u32 LE count][u32 LE sample_rate][count * f32 LE samples]
          count == 0  -> errore (dettagli su stderr)
  stderr: log ed errori

Selezione modello:
  - $HANDY_TTS_MODEL_FILE: nome file .onnx specifico (impostato da Handy)
  - $KOKORO_MODEL_DIR:     cartella dei modelli (riusata anche da Piper)
Il config e' lo stesso path + ".json".

Dipendenza: pip install piper-tts
"""

import glob
import json
import os
import struct
import sys

import numpy as np


def log(*a):
    print("[piper_tts_helper]", *a, file=sys.stderr, flush=True)


def find_model():
    """Trova il file .onnx Piper da caricare e il suo config .json."""
    model_dir = os.environ.get("KOKORO_MODEL_DIR") or "."
    forced = os.environ.get("HANDY_TTS_MODEL_FILE")
    onnx_path = None
    if forced:
        cand = os.path.join(model_dir, forced)
        if os.path.exists(cand):
            onnx_path = cand
        else:
            log(f"file forzato non trovato: {cand}, cerco un piper*.onnx")
    if onnx_path is None:
        # qualunque voce piper nella cartella
        matches = sorted(
            g for g in glob.glob(os.path.join(model_dir, "*.onnx"))
            if "kokoro" not in os.path.basename(g).lower()
        )
        if not matches:
            raise FileNotFoundError(
                f"nessuna voce Piper (.onnx) in {model_dir}")
        onnx_path = matches[0]
    config_path = onnx_path + ".json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config Piper mancante: {config_path}")
    return onnx_path, config_path


def main() -> int:
    try:
        from piper import PiperVoice
    except Exception as e:  # noqa: BLE001
        log(f"import piper fallito (pip install piper-tts?): {e}")
        return 1

    try:
        onnx_path, config_path = find_model()
        voice = PiperVoice.load(onnx_path, config_path=config_path)
        log(f"voce caricata: {os.path.basename(onnx_path)}")
    except Exception as e:  # noqa: BLE001
        log(f"caricamento voce fallito: {e}")
        return 1

    try:
        sample_rate = int(voice.config.sample_rate)
    except Exception:  # noqa: BLE001
        sample_rate = 22050

    stdout = sys.stdout.buffer

    def synth_int16(text: str, speed: float) -> bytes:
        """Restituisce PCM int16 LE. Gestisce piu' varianti dell'API piper."""
        length_scale = (1.0 / speed) if speed and speed > 0 else 1.0
        # API recenti: synthesize() -> iterabile di AudioChunk con int16 bytes
        if hasattr(voice, "synthesize"):
            try:
                from piper import SynthesisConfig  # type: ignore
                cfg = SynthesisConfig(length_scale=length_scale)
                chunks = voice.synthesize(text, syn_config=cfg)
            except Exception:  # noqa: BLE001 - firma diversa / no SynthesisConfig
                try:
                    chunks = voice.synthesize(text)
                except Exception:  # noqa: BLE001
                    chunks = None
            if chunks is not None:
                buf = bytearray()
                got = False
                for ch in chunks:
                    got = True
                    if hasattr(ch, "audio_int16_bytes"):
                        buf += ch.audio_int16_bytes
                    elif isinstance(ch, (bytes, bytearray)):
                        buf += ch
                if got:
                    return bytes(buf)
        # API piu' vecchie: synthesize_stream_raw() -> bytes int16
        if hasattr(voice, "synthesize_stream_raw"):
            buf = bytearray()
            try:
                for raw in voice.synthesize_stream_raw(text, length_scale=length_scale):
                    buf += raw
            except TypeError:
                for raw in voice.synthesize_stream_raw(text):
                    buf += raw
            return bytes(buf)
        raise RuntimeError("API piper non riconosciuta (synthesize/stream_raw assenti)")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            log("JSON non valido")
            stdout.write(struct.pack("<II", 0, sample_rate))
            stdout.flush()
            continue
        if req.get("cmd") == "quit":
            break
        text = (req.get("text") or "").strip()
        speed = float(req.get("speed", 1.0))
        if not text:
            stdout.write(struct.pack("<II", 0, sample_rate))
            stdout.flush()
            continue
        try:
            pcm16 = synth_int16(text, speed)
            samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
            pcm = np.ascontiguousarray(samples, dtype="<f4")
            stdout.write(struct.pack("<II", pcm.size, sample_rate))
            stdout.write(pcm.tobytes())
            stdout.flush()
        except Exception as e:  # noqa: BLE001
            log(f"sintesi fallita: {e}")
            stdout.write(struct.pack("<II", 0, sample_rate))
            stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

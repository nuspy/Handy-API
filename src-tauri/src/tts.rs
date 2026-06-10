//! Motore TTS Kokoro per Handy.
//!
//! Avvolge un helper Python persistente (`resources/kokoro_tts_helper.py`,
//! basato su `kokoro-onnx`). E' lo stesso pattern del subprocess `ffmpeg`
//! gia' usato in `api.rs`: isola il tooling Kokoro — acerbo in Rust e con
//! dipendenze native scomode (espeak-ng) — in un processo separato, e
//! mantiene il crate Handy snello.
//!
//! L'helper viene avviato una volta per sessione di chiamata e tenuto vivo,
//! cosi' il modello si carica una sola volta.

use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

use anyhow::{anyhow, Context, Result};
use log::info;
use serde_json::json;

/// Audio sintetizzato: campioni f32 mono alla frequenza nativa di Kokoro.
pub struct TtsAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Motore TTS: gestisce il processo helper Python e ci dialoga via pipe.
pub struct KokoroTts {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl KokoroTts {
    /// Avvia l'helper Python.
    ///
    /// - `helper_path`: percorso dell'helper Python (Kokoro o Piper)
    /// - `model_dir`: cartella dei modelli (passata come `KOKORO_MODEL_DIR`)
    /// - `model_file`: nome file specifico del modello da caricare (passato come
    ///   `HANDY_TTS_MODEL_FILE`); se `None` l'helper sceglie da solo.
    ///
    /// L'interprete e' `python`, sovrascrivibile con la env var `HANDY_PYTHON`.
    pub fn start(
        helper_path: &Path,
        model_dir: Option<&Path>,
        model_file: Option<&str>,
    ) -> Result<Self> {
        let python = std::env::var("HANDY_PYTHON").unwrap_or_else(|_| "python".to_string());

        let mut cmd = Command::new(&python);
        cmd.arg(helper_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // I log dell'helper finiscono nello stderr di Handy.
            .stderr(Stdio::inherit());
        if let Some(dir) = model_dir {
            cmd.env("KOKORO_MODEL_DIR", dir);
        }
        if let Some(file) = model_file {
            cmd.env("HANDY_TTS_MODEL_FILE", file);
        }
        #[cfg(target_os = "windows")]
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW

        let mut child = cmd.spawn().with_context(|| {
            format!("Avvio dell'helper Kokoro fallito ('{python}' non trovato?)")
        })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("stdin dell'helper Kokoro non disponibile"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("stdout dell'helper Kokoro non disponibile"))?;

        info!("KokoroTts: helper avviato ({})", helper_path.display());
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    /// Sintetizza `text`. **Operazione bloccante**: invocare da `spawn_blocking`.
    pub fn synthesize(
        &mut self,
        text: &str,
        voice: &str,
        lang: &str,
        speed: f32,
    ) -> Result<TtsAudio> {
        // 1. Invia la richiesta come riga JSON.
        let req = json!({"text": text, "voice": voice, "lang": lang, "speed": speed});
        writeln!(self.stdin, "{}", req)
            .context("scrittura su stdin dell'helper Kokoro fallita")?;
        self.stdin.flush().ok();

        // 2. Leggi l'header: [u32 LE count][u32 LE sample_rate].
        let mut header = [0u8; 8];
        self.stdout
            .read_exact(&mut header)
            .context("lettura header dall'helper Kokoro fallita (processo terminato?)")?;
        let count = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let sample_rate = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);

        if count == 0 {
            return Err(anyhow!(
                "sintesi Kokoro fallita (vedi i log dell'helper su stderr)"
            ));
        }

        // 3. Leggi `count` campioni f32 little-endian.
        let mut raw = vec![0u8; count * 4];
        self.stdout
            .read_exact(&mut raw)
            .context("lettura dei campioni dall'helper Kokoro fallita")?;
        let samples: Vec<f32> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        Ok(TtsAudio {
            samples,
            sample_rate,
        })
    }
}

impl Drop for KokoroTts {
    fn drop(&mut self) {
        // Chiedi all'helper di uscire con grazia, poi assicurati che muoia.
        let _ = writeln!(self.stdin, "{}", json!({"cmd": "quit"}));
        let _ = self.stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

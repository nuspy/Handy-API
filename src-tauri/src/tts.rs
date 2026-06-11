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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

use anyhow::{anyhow, Context, Result};
use log::{error, info};
use serde_json::json;

/// Tetto per una singola sintesi: oltre, l'helper e' considerato BLOCCATO e
/// viene ucciso dal watchdog (le read/write sulla pipe falliscono e il
/// chiamante riceve Err invece di restare appeso per sempre col mutex
/// dell'engine preso). L'engine morto viene poi scartato da
/// `CallModelManager::reap_dead_tts` e ricaricato al `/models/load` successivo.
const SYNTH_TIMEOUT: Duration = Duration::from_secs(60);

/// Audio sintetizzato: campioni f32 mono alla frequenza nativa di Kokoro.
pub struct TtsAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Motore TTS: gestisce il processo helper Python e ci dialoga via pipe.
pub struct KokoroTts {
    /// Condiviso col thread watchdog (che deve poterlo uccidere su timeout).
    child: Arc<Mutex<Child>>,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

/// Disattiva il watchdog quando la sintesi esce (per qualsiasi via).
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

impl KokoroTts {
    /// Avvia l'helper Python.
    ///
    /// - `helper_path`: percorso dell'helper Python (Kokoro, Piper o Chatterbox)
    /// - `model_dir`: cartella dei modelli (passata come `KOKORO_MODEL_DIR`)
    /// - `model_file`: nome file specifico del modello da caricare (passato come
    ///   `HANDY_TTS_MODEL_FILE`); se `None` l'helper sceglie da solo.
    /// - `python_override`: interprete specifico per questo engine (es. il venv
    ///   dedicato di Chatterbox); `None` = `HANDY_PYTHON` o `python` dal PATH.
    pub fn start(
        helper_path: &Path,
        model_dir: Option<&Path>,
        model_file: Option<&str>,
        python_override: Option<&Path>,
    ) -> Result<Self> {
        let python = python_override
            .map(|p| p.to_string_lossy().into_owned())
            .or_else(|| std::env::var("HANDY_PYTHON").ok())
            .unwrap_or_else(|| "python".to_string());

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
            child: Arc::new(Mutex::new(child)),
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    /// True se il processo helper e' ancora vivo.
    pub fn is_alive(&self) -> bool {
        self.child
            .lock()
            .map(|mut c| matches!(c.try_wait(), Ok(None)))
            .unwrap_or(false)
    }

    /// Sintetizza `text`. **Operazione bloccante**: invocare da `spawn_blocking`.
    ///
    /// Protetta da un watchdog: se l'helper non risponde entro
    /// [`SYNTH_TIMEOUT`] viene ucciso, cosi' la lettura fallisce subito invece
    /// di bloccare per sempre il mutex dell'engine.
    pub fn synthesize(
        &mut self,
        text: &str,
        voice: &str,
        lang: &str,
        speed: f32,
    ) -> Result<TtsAudio> {
        self.synthesize_with_timeout(text, voice, lang, speed, SYNTH_TIMEOUT)
    }

    /// Come [`Self::synthesize`] ma con un watchdog esplicito: il warm-up al
    /// load (Chatterbox carica il modello al primo turno, anche minuti) e
    /// /tts/test usano timeout piu' generosi del default da streaming.
    pub fn synthesize_with_timeout(
        &mut self,
        text: &str,
        voice: &str,
        lang: &str,
        speed: f32,
        timeout: Duration,
    ) -> Result<TtsAudio> {
        // Watchdog: parte PRIMA della write (anche la write puo' bloccare se
        // l'helper non legge piu' e la pipe e' piena).
        let cancel = Arc::new(AtomicBool::new(false));
        let _guard = CancelOnDrop(cancel.clone());
        {
            let cancel = cancel.clone();
            let child = self.child.clone();
            std::thread::Builder::new()
                .name("tts-watchdog".to_string())
                .spawn(move || {
                    let deadline = Instant::now() + timeout;
                    while Instant::now() < deadline {
                        if cancel.load(Ordering::Relaxed) {
                            return;
                        }
                        std::thread::sleep(Duration::from_millis(100));
                    }
                    if !cancel.load(Ordering::Relaxed) {
                        error!(
                            "KokoroTts: helper bloccato oltre {}s, lo termino",
                            timeout.as_secs()
                        );
                        if let Ok(mut c) = child.lock() {
                            let _ = c.kill();
                        }
                    }
                })
                .ok();
        }

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
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

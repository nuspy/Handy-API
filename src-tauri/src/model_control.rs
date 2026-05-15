//! Canale di controllo + lifecycle dei modelli per la sessione di chiamata.
//!
//! L'orchestrator chiede a Handy di caricare/scaricare i modelli in VRAM e
//! interroga le voci/lingue disponibili. I modelli restano in memoria e si
//! auto-scaricano dopo 15 minuti di inutilizzo.
//!
//! Endpoint HTTP (request/response, non WebSocket):
//!   GET  /models/available  -> voci/lingue TTS + stato modello STT
//!   POST /models/load       -> carica {stt_model?, tts?} in VRAM
//!   POST /models/unload     -> scarica {component: "stt"|"tts"|"all"}
//!   GET  /models/status     -> cosa e' caricato + inattivita'

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::{extract::State, Json};
use log::{error, info};
use serde_json::{json, Value};

use crate::api::ApiState;
use crate::managers::transcription::TranscriptionManager;
use crate::tts::KokoroTts;

/// Voci Kokoro v1.0 con la rispettiva lingua. Usato da `/models/available` e
/// dalla logica "prefisso internazionale -> lingua" dell'orchestrator.
pub const KOKORO_VOICES: &[(&str, &str, &str)] = &[
    ("if_sara", "it", "Italiano"),
    ("im_nicola", "it", "Italiano"),
    ("af_heart", "en-us", "English (US)"),
    ("am_michael", "en-us", "English (US)"),
    ("bf_emma", "en-gb", "English (UK)"),
    ("bm_george", "en-gb", "English (UK)"),
    ("ff_siwis", "fr-fr", "Francais"),
    ("ef_dora", "es", "Espanol"),
    ("em_alex", "es", "Espanol"),
    ("pf_dora", "pt-br", "Portugues"),
    ("pm_alex", "pt-br", "Portugues"),
    ("jf_alpha", "ja", "Nihongo"),
    ("jm_kumo", "ja", "Nihongo"),
    ("zf_xiaoxiao", "cmn", "Zhongwen"),
    ("zm_yunxi", "cmn", "Zhongwen"),
    ("hf_alpha", "hi", "Hindi"),
    ("hm_omega", "hi", "Hindi"),
];

/// Dopo quanta inattivita' un modello viene scaricato da solo.
const IDLE_UNLOAD: Duration = Duration::from_secs(15 * 60);

/// Gestisce il caricamento/scaricamento dei modelli STT e TTS in VRAM.
pub struct CallModelManager {
    transcription: Arc<TranscriptionManager>,
    tts_helper_path: PathBuf,
    tts_model_dir: PathBuf,
    inner: Mutex<Inner>,
}

struct Inner {
    /// Engine TTS Kokoro caricato (`None` = non in VRAM).
    tts: Option<Arc<Mutex<KokoroTts>>>,
    /// Ultimo utilizzo (STT o TTS), per l'auto-unload.
    last_used: Instant,
}

impl CallModelManager {
    pub fn new(
        transcription: Arc<TranscriptionManager>,
        tts_helper_path: PathBuf,
        tts_model_dir: PathBuf,
    ) -> Self {
        Self {
            transcription,
            tts_helper_path,
            tts_model_dir,
            inner: Mutex::new(Inner {
                tts: None,
                last_used: Instant::now(),
            }),
        }
    }

    /// Aggiorna il timestamp di utilizzo (chiamato da `/stt/stream` e `/tts/stream`).
    pub fn touch(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.last_used = Instant::now();
        }
    }

    /// Ritorna l'engine TTS se caricato in VRAM, altrimenti `None`.
    pub fn tts_engine(&self) -> Option<Arc<Mutex<KokoroTts>>> {
        self.inner.lock().ok().and_then(|inner| inner.tts.clone())
    }

    /// Carica i modelli richiesti in VRAM. **Bloccante**: usare da `spawn_blocking`.
    pub fn load(&self, stt_model: Option<&str>, load_tts: bool) -> Result<(), String> {
        // --- STT (Whisper / Parakeet) ---
        match stt_model {
            Some(model_id) => self
                .transcription
                .load_model(model_id)
                .map_err(|e| format!("STT load fallito: {e}"))?,
            // Nessun modello indicato: assicura che almeno il default sia pronto.
            None => self.transcription.initiate_model_load(),
        }

        // --- TTS (Kokoro) ---
        if load_tts {
            let mut inner = self.inner.lock().map_err(|_| "lock avvelenato".to_string())?;
            if inner.tts.is_none() {
                let engine =
                    KokoroTts::start(&self.tts_helper_path, Some(self.tts_model_dir.as_path()))
                        .map_err(|e| format!("TTS load fallito: {e}"))?;
                inner.tts = Some(Arc::new(Mutex::new(engine)));
            }
            inner.last_used = Instant::now();
        }
        Ok(())
    }

    /// Scarica dalla VRAM il componente indicato (`"stt"`, `"tts"` o `"all"`).
    pub fn unload(&self, component: &str) -> Result<(), String> {
        match component {
            "stt" => self
                .transcription
                .unload_model()
                .map_err(|e| format!("STT unload fallito: {e}"))?,
            "tts" => {
                if let Ok(mut inner) = self.inner.lock() {
                    inner.tts = None; // Drop di KokoroTts -> termina l'helper Python
                }
            }
            "all" => {
                let _ = self.transcription.unload_model();
                if let Ok(mut inner) = self.inner.lock() {
                    inner.tts = None;
                }
            }
            other => return Err(format!("componente sconosciuto: {other}")),
        }
        Ok(())
    }

    /// Stato corrente: cosa e' caricato e da quanto e' inattivo.
    pub fn status(&self) -> Value {
        let (tts_loaded, idle_secs) = match self.inner.lock() {
            Ok(inner) => (inner.tts.is_some(), inner.last_used.elapsed().as_secs()),
            Err(_) => (false, 0),
        };
        json!({
            "stt": {
                "loaded": self.transcription.is_model_loaded(),
                "current_model": self.transcription.get_current_model(),
            },
            "tts": { "loaded": tts_loaded },
            "idle_seconds": idle_secs,
            "idle_unload_seconds": IDLE_UNLOAD.as_secs(),
        })
    }

    /// Controlla l'inattivita' e scarica i modelli se oltre la soglia.
    fn check_idle(&self) {
        let expired = match self.inner.lock() {
            Ok(inner) => {
                let loaded = inner.tts.is_some() || self.transcription.is_model_loaded();
                loaded && inner.last_used.elapsed() >= IDLE_UNLOAD
            }
            Err(_) => false,
        };
        if expired {
            info!("Modelli inattivi da oltre 15 minuti: scarico dalla VRAM");
            if let Err(e) = self.unload("all") {
                error!("Auto-unload fallito: {}", e);
            }
        }
    }
}

/// Avvia il thread che scarica i modelli dopo 15 minuti di inattivita'.
pub fn spawn_idle_unloader(manager: Arc<CallModelManager>) {
    std::thread::Builder::new()
        .name("model-idle-unloader".to_string())
        .spawn(move || loop {
            std::thread::sleep(Duration::from_secs(60));
            manager.check_idle();
        })
        .expect("impossibile avviare il thread di idle-unload");
}

// ============================================================================
//  Handler HTTP
// ============================================================================

/// `GET /models/available` — voci/lingue TTS e stato del modello STT.
pub async fn models_available(State(state): State<Arc<ApiState>>) -> Json<Value> {
    let voices: Vec<Value> = KOKORO_VOICES
        .iter()
        .map(|(voice, lang, lang_name)| {
            json!({"voice": voice, "lang": lang, "language": lang_name})
        })
        .collect();
    Json(json!({
        "tts": { "voices": voices },
        "stt": {
            "loaded": state.transcription_manager.is_model_loaded(),
            "current_model": state.transcription_manager.get_current_model(),
        },
    }))
}

/// `POST /models/load` — carica i modelli in VRAM.
/// Body: `{"stt_model": "<id>"?, "tts": true|false}` (tts default true).
pub async fn models_load(
    State(state): State<Arc<ApiState>>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let stt_model = body
        .get("stt_model")
        .and_then(Value::as_str)
        .map(String::from);
    let load_tts = body.get("tts").and_then(Value::as_bool).unwrap_or(true);

    let manager = state.call_models.clone();
    let result = tokio::task::spawn_blocking(move || {
        manager.load(stt_model.as_deref(), load_tts)
    })
    .await;

    match result {
        Ok(Ok(())) => Json(json!({"status": "ready"})),
        Ok(Err(e)) => Json(json!({"status": "error", "message": e})),
        Err(e) => Json(json!({"status": "error", "message": format!("task panicked: {e}")})),
    }
}

/// `POST /models/unload` — scarica un componente dalla VRAM.
/// Body: `{"component": "stt"|"tts"|"all"}` (default "all").
pub async fn models_unload(
    State(state): State<Arc<ApiState>>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let component = body
        .get("component")
        .and_then(Value::as_str)
        .unwrap_or("all")
        .to_string();
    match state.call_models.unload(&component) {
        Ok(()) => Json(json!({"status": "ok", "unloaded": component})),
        Err(e) => Json(json!({"status": "error", "message": e})),
    }
}

/// `GET /models/status` — cosa e' caricato e da quanto e' inattivo.
pub async fn models_status(State(state): State<Arc<ApiState>>) -> Json<Value> {
    Json(state.call_models.status())
}

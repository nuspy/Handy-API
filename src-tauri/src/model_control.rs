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

use axum::{
    extract::{Path as AxumPath, State},
    http::StatusCode,
    Json,
};
use log::{error, info};
use serde_json::{json, Value};

use crate::api::ApiState;
use crate::managers::model::{EngineType, ModelManager};
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

/// Watchdog del warm-up al load dell'engine: Chatterbox carica il modello al
/// primo turno (from_pretrained: misurati ~3-4 minuti su questa macchina coi
/// pesi gia' in cache HF) + la sintesi della frase di prova. Molto oltre il
/// SYNTH_TIMEOUT da streaming (60s).
const WARMUP_TIMEOUT: Duration = Duration::from_secs(600);

/// Watchdog delle sintesi one-shot di /tts/test: testo libero + eventuale
/// preparazione dei conditionals di una voce clonata.
const TTS_TEST_TIMEOUT: Duration = Duration::from_secs(300);

/// Gestisce il caricamento/scaricamento dei modelli STT e TTS in VRAM.
pub struct CallModelManager {
    transcription: Arc<TranscriptionManager>,
    model_manager: Arc<ModelManager>,
    app_handle: tauri::AppHandle,
    /// Helper Kokoro (le risorse Piper stanno nella stessa cartella).
    tts_helper_path: PathBuf,
    tts_model_dir: PathBuf,
    inner: Mutex<Inner>,
}

struct Inner {
    /// Engine TTS caricato (`None` = non in VRAM).
    tts: Option<Arc<Mutex<KokoroTts>>>,
    /// Id del modello TTS caricato (per ricaricare se l'utente lo cambia).
    tts_model: Option<String>,
    /// Ultimo utilizzo (STT o TTS), per l'auto-unload.
    last_used: Instant,
}

impl CallModelManager {
    pub fn new(
        transcription: Arc<TranscriptionManager>,
        model_manager: Arc<ModelManager>,
        app_handle: tauri::AppHandle,
        tts_helper_path: PathBuf,
        tts_model_dir: PathBuf,
    ) -> Self {
        Self {
            transcription,
            model_manager,
            app_handle,
            tts_helper_path,
            tts_model_dir,
            inner: Mutex::new(Inner {
                tts: None,
                tts_model: None,
                last_used: Instant::now(),
            }),
        }
    }

    /// Id del modello TTS scelto nei settings (Settings -> Models -> TTS).
    pub fn selected_tts_model(&self) -> String {
        crate::settings::get_settings(&self.app_handle).selected_tts_model
    }

    /// Voce TTS preferita nei settings ("" = default dell'engine).
    pub fn selected_tts_voice(&self) -> String {
        crate::settings::get_settings(&self.app_handle).selected_tts_voice
    }

    /// Percorso di un helper che vive accanto a quello Kokoro.
    fn helper_sibling(&self, name: &str) -> PathBuf {
        self.tts_helper_path
            .parent()
            .map(|p| p.join(name))
            .unwrap_or_else(|| self.tts_helper_path.clone())
    }

    /// Python del venv dedicato a Chatterbox (creato con --system-site-packages
    /// per riusare il torch CUDA di sistema, con le dipendenze pinnate di
    /// chatterbox isolate). Override con HANDY_PYTHON_CHATTERBOX.
    fn chatterbox_python(&self) -> Option<PathBuf> {
        if let Ok(p) = std::env::var("HANDY_PYTHON_CHATTERBOX") {
            return Some(PathBuf::from(p));
        }
        let venv = self
            .tts_model_dir
            .parent()?
            .join("chatterbox-venv")
            .join("Scripts")
            .join("python.exe");
        venv.is_file().then_some(venv)
    }

    /// Cartella dei campioni voce per il cloning (POST /voices/clone).
    pub fn cloned_voices_dir(&self) -> PathBuf {
        self.tts_model_dir.join("cloned_voices")
    }

    /// Elenco delle voci clonate disponibili (nomi senza estensione).
    pub fn cloned_voices(&self) -> Vec<String> {
        let mut out = Vec::new();
        if let Ok(entries) = std::fs::read_dir(self.cloned_voices_dir()) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_file()
                    && matches!(
                        p.extension().and_then(|x| x.to_str()),
                        Some("wav") | Some("mp3") | Some("flac") | Some("ogg")
                    )
                {
                    if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                        out.push(stem.to_string());
                    }
                }
            }
        }
        out.sort();
        out
    }

    /// Id del modello Chatterbox SCARICATO, se presente (serve a /tts/test
    /// per provare le voci clonate anche quando l'engine attivo e' un altro).
    pub fn chatterbox_model_id(&self) -> Option<String> {
        self.model_manager
            .get_available_models()
            .into_iter()
            .find(|m| matches!(m.engine_type, EngineType::Chatterbox) && m.is_downloaded)
            .map(|m| m.id)
    }

    /// Risolve il modello TTS attivo (dai settings) in
    /// (helper_path, model_file, model_id, python_override). Gli helper
    /// condividono lo stesso protocollo stdio: cambia lo script, l'eventuale
    /// file modello e (per Chatterbox) l'interprete del venv dedicato.
    /// Se il modello non e' noto, ritorna l'helper Kokoro senza file
    /// specifico (l'helper sceglie da solo).
    fn resolve_tts_id(&self, sel: &str) -> (PathBuf, Option<String>, String, Option<PathBuf>) {
        let sel = sel.to_string();
        match self.model_manager.get_model_info(&sel) {
            Some(info) if matches!(info.engine_type, EngineType::Piper) => (
                self.helper_sibling("piper_tts_helper.py"),
                Some(info.filename),
                sel,
                None,
            ),
            Some(info) if matches!(info.engine_type, EngineType::Chatterbox) => (
                self.helper_sibling("chatterbox_tts_helper.py"),
                None, // i pesi li gestisce l'helper (cache HuggingFace)
                sel,
                self.chatterbox_python(),
            ),
            Some(info) => (self.tts_helper_path.clone(), Some(info.filename), sel, None),
            None => (self.tts_helper_path.clone(), None, sel, None),
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

        // --- TTS (Kokoro/Piper/Chatterbox, da settings.selected_tts_model) ---
        if load_tts {
            self.ensure_tts()?;
        }
        Ok(())
    }

    /// Carica il SOLO engine TTS se non e' gia' in VRAM col modello selezionato
    /// nei settings. Non tocca l'STT (usato da `/tts/test` oltre che da `load`).
    /// **Bloccante**: usare da `spawn_blocking`.
    pub fn ensure_tts(&self) -> Result<(), String> {
        self.ensure_tts_model(None)
    }

    /// Come `ensure_tts`, ma con un modello esplicito al posto di quello dei
    /// settings: /tts/test la usa per provare le voci clonate (engine
    /// Chatterbox) senza cambiare la selezione dell'utente. Il prossimo
    /// /models/load ricarica l'engine selezionato.
    pub fn ensure_tts_model(&self, model_id_override: Option<&str>) -> Result<(), String> {
        let sel = model_id_override
            .map(String::from)
            .unwrap_or_else(|| self.selected_tts_model());
        let (helper, model_file, model_id, python) = self.resolve_tts_id(&sel);
        let mut inner = self.inner.lock().map_err(|_| "lock avvelenato".to_string())?;
        // Ricarica anche se gia' caricato MA con un modello diverso da
        // quello ora selezionato nei settings (l'utente l'ha cambiato).
        let needs_load = inner.tts.is_none()
            || inner.tts_model.as_deref() != Some(model_id.as_str());
        if needs_load {
            if inner.tts.is_some() {
                info!(
                    "TTS: modello cambiato ({} -> {}), ricarico l'engine",
                    inner.tts_model.as_deref().unwrap_or("?"),
                    model_id
                );
            }
            inner.tts = None; // drop del vecchio engine -> termina l'helper
            inner.tts_model = None;
            let mut engine = KokoroTts::start(
                &helper,
                Some(self.tts_model_dir.as_path()),
                model_file.as_deref(),
                python.as_deref(),
            )
            .map_err(|e| format!("TTS load fallito: {e}"))?;
            // Warm-up: una micro-sintesi forza il caricamento COMPLETO del
            // modello dentro il load (timeout dedicato e generoso). Senza,
            // la prima sintesi reale pagherebbe il load dell'helper e il
            // watchdog da 60s la ucciderebbe (visto con Chatterbox: ~60-120s
            // di from_pretrained al primo turno).
            let is_chatterbox = helper
                .file_name()
                .and_then(|s| s.to_str())
                .map_or(false, |s| s.contains("chatterbox"));
            // Chatterbox: voce builtin (""). Kokoro: serve una voce valida.
            // Piper: ignora il campo voice.
            let warm_voice = if is_chatterbox { "" } else { "if_sara" };
            // NB: frase di lunghezza normale — Chatterbox va in errore
            // tensoriale sui testi cortissimi ("Ok." -> reduction dim vuota).
            engine
                .synthesize_with_timeout(
                    "Questo e' il riscaldamento del motore di sintesi vocale.",
                    warm_voice,
                    "it",
                    1.0,
                    WARMUP_TIMEOUT,
                )
                .map_err(|e| format!("warm-up TTS fallito: {e}"))?;
            info!("TTS: engine {} pronto (warm-up ok)", model_id);
            inner.tts = Some(Arc::new(Mutex::new(engine)));
            inner.tts_model = Some(model_id);
        }
        inner.last_used = Instant::now();
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
                    inner.tts_model = None;
                }
            }
            "all" => {
                let _ = self.transcription.unload_model();
                if let Ok(mut inner) = self.inner.lock() {
                    inner.tts = None;
                    inner.tts_model = None;
                }
            }
            other => return Err(format!("componente sconosciuto: {other}")),
        }
        Ok(())
    }

    /// Stato corrente: cosa e' caricato e da quanto e' inattivo.
    pub fn status(&self) -> Value {
        let (tts_loaded, tts_model, idle_secs) = match self.inner.lock() {
            Ok(inner) => (
                inner.tts.is_some(),
                inner.tts_model.clone(),
                inner.last_used.elapsed().as_secs(),
            ),
            Err(_) => (false, None, 0),
        };
        json!({
            "stt": {
                "loaded": self.transcription.is_model_loaded(),
                "current_model": self.transcription.get_current_model(),
            },
            "tts": {
                "loaded": tts_loaded,
                "current_model": tts_model,
                "selected_model": self.selected_tts_model(),
            },
            "idle_seconds": idle_secs,
            "idle_unload_seconds": IDLE_UNLOAD.as_secs(),
        })
    }

    /// Se l'engine TTS e' MORTO (helper ucciso dal watchdog su timeout, o
    /// crashato), lo scarta: cosi' il prossimo `/models/load` lo ricarica
    /// pulito invece di restituire un engine inutilizzabile. Un mutex
    /// avvelenato (panic durante una sintesi) vale come morto.
    pub fn reap_dead_tts(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            let dead = inner
                .tts
                .as_ref()
                .map_or(false, |t| t.lock().map(|e| !e.is_alive()).unwrap_or(true));
            if dead {
                error!("TTS: helper morto, scarto l'engine (si ricarica al prossimo load)");
                inner.tts = None;
                inner.tts_model = None;
            }
        }
    }

    /// Controlla l'inattivita' e scarica il TTS se oltre la soglia.
    ///
    /// NB: gestiamo SOLO il TTS (Kokoro), che e' di proprieta' esclusiva del
    /// CallModelManager. Il modello STT ha il proprio idle-watcher dentro
    /// `TranscriptionManager`, che aggiorna `last_activity` ad ogni trascrizione
    /// (sia da chiamata che da dettatura desktop) e salta l'unload mentre si
    /// registra. Se qui includessimo l'STT, scaricheremmo ogni 60s il modello
    /// che la dettatura desktop ha appena caricato — `last_used` non viene mai
    /// rinfrescato dalla dettatura (solo da /stt/stream e /tts/stream) — facendo
    /// fallire la trascrizione con "Model is not loaded".
    fn check_idle(&self) {
        let expired = match self.inner.lock() {
            Ok(inner) => inner.tts.is_some() && inner.last_used.elapsed() >= IDLE_UNLOAD,
            Err(_) => false,
        };
        if expired {
            info!("TTS inattivo da oltre 15 minuti: scarico dalla VRAM");
            if let Err(e) = self.unload("tts") {
                error!("Auto-unload TTS fallito: {}", e);
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
///
/// Le voci dipendono dall'engine del modello TTS selezionato:
/// - Kokoro: le voci interne del modello (KOKORO_VOICES), selezionabili per
///   richiesta col campo "voice" di /tts/stream;
/// - Piper: la voce E' il modello — si elencano le voci Piper SCARICATE
///   (voice = model id); per cambiarla si imposta selected_tts_model e si
///   rifa' /models/load (l'engine viene ricaricato). Il campo "voice" di
///   /tts/stream viene ignorato dall'helper Piper;
/// - Chatterbox: le voci sono i campioni CLONATI (POST /voices/clone); il
///   campo "voice" di /tts/stream seleziona il campione per-richiesta.
pub async fn models_available(State(state): State<Arc<ApiState>>) -> Json<Value> {
    let selected = state.call_models.selected_tts_model();
    let engine = state
        .model_manager
        .get_model_info(&selected)
        .map(|i| i.engine_type);
    let (engine_name, voices): (&str, Vec<Value>) = match engine {
        Some(EngineType::Piper) => (
            "piper",
            state
                .model_manager
                .get_available_models()
                .into_iter()
                .filter(|m| matches!(m.engine_type, EngineType::Piper) && m.is_downloaded)
                .map(|m| {
                    json!({
                        "voice": m.id,
                        "lang": m.supported_languages.first().cloned().unwrap_or_default(),
                        "language": m.name,
                    })
                })
                .collect(),
        ),
        Some(EngineType::Chatterbox) => (
            "chatterbox",
            state
                .call_models
                .cloned_voices()
                .into_iter()
                .map(|v| json!({"voice": v, "lang": "*", "language": "Voce clonata"}))
                .collect(),
        ),
        _ => (
            "kokoro",
            KOKORO_VOICES
                .iter()
                .map(|(voice, lang, lang_name)| {
                    json!({"voice": voice, "lang": lang, "language": lang_name})
                })
                .collect(),
        ),
    };
    Json(json!({
        "tts": {
            "engine": engine_name,
            "voices": voices,
            "selected_model": selected,
        },
        "stt": {
            "loaded": state.transcription_manager.is_model_loaded(),
            "current_model": state.transcription_manager.get_current_model(),
        },
    }))
}

/// `POST /voices/clone` — salva un campione voce per il cloning Chatterbox.
/// Body: `{"name": "<slug>", "audio_b64": "<base64>", "format": "wav"?}`.
/// 5-15 secondi di parlato pulito bastano. La voce diventa selezionabile come
/// campo "voice" di /tts/stream (con l'engine Chatterbox attivo).
pub async fn voices_clone(
    State(state): State<Arc<ApiState>>,
    Json(body): Json<Value>,
) -> Json<Value> {
    use base64::Engine as _;

    let name = body.get("name").and_then(Value::as_str).unwrap_or("").trim();
    let safe: String = name
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    if safe.is_empty() {
        return Json(json!({"status": "error",
                           "message": "campo 'name' mancante o non valido"}));
    }
    let b64 = body.get("audio_b64").and_then(Value::as_str).unwrap_or("");
    let bytes = match base64::engine::general_purpose::STANDARD.decode(b64) {
        Ok(b) if !b.is_empty() => b,
        _ => {
            return Json(json!({"status": "error",
                               "message": "campo 'audio_b64' mancante o non decodificabile"}));
        }
    };
    let ext = match body.get("format").and_then(Value::as_str).unwrap_or("wav") {
        f @ ("wav" | "mp3" | "flac" | "ogg") => f,
        other => {
            return Json(json!({"status": "error",
                               "message": format!("formato non supportato: {other}")}));
        }
    };
    let dir = state.call_models.cloned_voices_dir();
    if let Err(e) = std::fs::create_dir_all(&dir) {
        return Json(json!({"status": "error", "message": format!("mkdir fallita: {e}")}));
    }
    let path = dir.join(format!("{safe}.{ext}"));
    if let Err(e) = std::fs::write(&path, &bytes) {
        return Json(json!({"status": "error", "message": format!("scrittura fallita: {e}")}));
    }
    info!("Voce clonata salvata: {} ({} byte)", path.display(), bytes.len());
    Json(json!({"status": "ok", "voice": safe,
                "voices": state.call_models.cloned_voices()}))
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

/// `GET /voices/cloned` — elenco delle voci clonate con i metadata dei file.
/// A differenza di `/models/available` funziona a prescindere dall'engine
/// TTS attivo (la GUI gestisce le voci anche con Kokoro/Piper selezionato).
pub async fn voices_list(State(state): State<Arc<ApiState>>) -> Json<Value> {
    let dir = state.call_models.cloned_voices_dir();
    let mut voices: Vec<Value> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for e in entries.flatten() {
            let p = e.path();
            let ext = p
                .extension()
                .and_then(|x| x.to_str())
                .unwrap_or("")
                .to_string();
            if !p.is_file() || !matches!(ext.as_str(), "wav" | "mp3" | "flac" | "ogg") {
                continue;
            }
            let name = match p.file_stem().and_then(|s| s.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            let meta = e.metadata().ok();
            let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
            let modified = meta
                .and_then(|m| m.modified().ok())
                .map(|t| chrono::DateTime::<chrono::Utc>::from(t).to_rfc3339())
                .unwrap_or_default();
            voices.push(json!({
                "name": name,
                "format": ext,
                "size_bytes": size,
                "modified": modified,
            }));
        }
    }
    voices.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    Json(json!({"voices": voices}))
}

/// `DELETE /voices/cloned/:name` — elimina un campione voce clonata.
/// Il nome viene sanificato con lo stesso filtro di `/voices/clone`.
pub async fn voices_delete(
    State(state): State<Arc<ApiState>>,
    AxumPath(name): AxumPath<String>,
) -> (StatusCode, Json<Value>) {
    let safe: String = name
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    if safe.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"status": "error", "message": "nome voce non valido"})),
        );
    }
    let dir = state.call_models.cloned_voices_dir();
    let mut removed = false;
    for ext in ["wav", "mp3", "flac", "ogg"] {
        let path = dir.join(format!("{safe}.{ext}"));
        if path.is_file() && std::fs::remove_file(&path).is_ok() {
            info!("Voce clonata eliminata: {}", path.display());
            removed = true;
        }
    }
    if removed {
        (
            StatusCode::OK,
            Json(json!({"status": "ok", "voices": state.call_models.cloned_voices()})),
        )
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(json!({"status": "error",
                        "message": format!("voce '{safe}' non trovata")})),
        )
    }
}

/// `POST /tts/test` — sintesi one-shot per provare una voce (anteprima GUI).
/// Body: `{"text", "voice"?, "lang"?, "speed"?}`. Se `voice` manca usa la
/// preferita dei settings (`selected_tts_voice`). Carica il solo engine TTS
/// se non e' gia' in VRAM.
/// Risposta: `{"status":"ok","audio_b64":"<wav>","sample_rate":N,"duration_ms":N}`.
pub async fn tts_test(
    State(state): State<Arc<ApiState>>,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    let text = body
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if text.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"status": "error", "message": "campo 'text' mancante"})),
        );
    }
    let voice = body
        .get("voice")
        .and_then(Value::as_str)
        .filter(|v| !v.is_empty())
        .map(String::from)
        .unwrap_or_else(|| {
            let preferred = state.call_models.selected_tts_voice();
            if preferred.is_empty() {
                "if_sara".to_string()
            } else {
                preferred
            }
        });
    let lang = body
        .get("lang")
        .and_then(Value::as_str)
        .unwrap_or("it")
        .to_string();
    let speed = body.get("speed").and_then(Value::as_f64).unwrap_or(1.0) as f32;

    // Voce clonata? Serve l'engine Chatterbox, a prescindere dal modello
    // selezionato nei settings (es. Kokoro attivo): altrimenti l'helper
    // attivo non conosce la voce e la sintesi fallisce.
    let is_cloned = state
        .call_models
        .cloned_voices()
        .iter()
        .any(|v| v == &voice);
    let model_override = if is_cloned {
        match state.call_models.chatterbox_model_id() {
            Some(id) => Some(id),
            None => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({"status": "error",
                                "message": "voce clonata ma il modello Chatterbox non \
                                            e' scaricato (Settings -> Models -> TTS)"})),
                );
            }
        }
    } else {
        None
    };

    // Se l'helper e' morto (watchdog/crash) scartalo, cosi' lo ricarichiamo.
    state.call_models.reap_dead_tts();
    let manager = state.call_models.clone();
    match tokio::task::spawn_blocking(move || {
        manager.ensure_tts_model(model_override.as_deref())
    })
    .await
    {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"status": "error", "message": e})),
            );
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"status": "error", "message": format!("task panicked: {e}")})),
            );
        }
    }
    let engine = match state.call_models.tts_engine() {
        Some(e) => e,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"status": "error", "message": "engine TTS non caricato"})),
            );
        }
    };
    state.call_models.touch();

    let voice_used = voice.clone();
    let synth = tokio::task::spawn_blocking(move || {
        let mut eng = engine
            .lock()
            .map_err(|_| "lock dell'engine avvelenato".to_string())?;
        eng.synthesize_with_timeout(&text, &voice, &lang, speed, TTS_TEST_TIMEOUT)
            .map_err(|e| e.to_string())
    })
    .await;
    let audio = match synth {
        Ok(Ok(a)) => a,
        Ok(Err(e)) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"status": "error", "message": e})),
            );
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"status": "error", "message": format!("task panicked: {e}")})),
            );
        }
    };

    // PCM f32 -> WAV 16 bit mono in memoria.
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: audio.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = match hound::WavWriter::new(&mut cursor, spec) {
            Ok(w) => w,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"status": "error",
                                "message": format!("encoding WAV fallito: {e}")})),
                );
            }
        };
        for s in &audio.samples {
            let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            if writer.write_sample(v).is_err() {
                break;
            }
        }
        if let Err(e) = writer.finalize() {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"status": "error",
                            "message": format!("finalizzazione WAV fallita: {e}")})),
            );
        }
    }
    use base64::Engine as _;
    let b64 = base64::engine::general_purpose::STANDARD.encode(cursor.into_inner());
    let duration_ms =
        audio.samples.len() as u64 * 1000 / u64::from(audio.sample_rate.max(1));
    (
        StatusCode::OK,
        Json(json!({
            "status": "ok",
            "audio_b64": b64,
            "sample_rate": audio.sample_rate,
            "duration_ms": duration_ms,
            "voice": voice_used,
        })),
    )
}

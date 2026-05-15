//! WebSocket di streaming per la "sessione di chiamata".
//!
//! Espone gli endpoint che il call-bridge usa durante una telefonata:
//!
//! - `GET /stt/stream` — il call-bridge invia PCM continuo (f32 little-endian,
//!   mono, 16 kHz). Handy fa VAD (Silero smoothed) e, a fine enunciato,
//!   trascrive e rimanda il testo come JSON.
//!
//! - `GET /tts/stream` — il call-bridge invia il testo di risposta di Hermes
//!   a pezzi (mentre Hermes lo genera). Handy lo accumula e, a ogni confine
//!   di frase, sintetizza con Kokoro e rimanda subito l'audio: lo streaming
//!   parte dopo la prima frase, non a fine risposta.
//!
//! ## Protocollo `/stt/stream`
//!
//! Client -> server:
//!   - messaggi **binari**: PCM `f32` LE mono a 16 kHz (qualsiasi lunghezza
//!     multipla di 4 byte). Il call-bridge ricampiona a monte se il device
//!     HFP gira a 8 kHz.
//!   - messaggi **testo**: riservati a comandi di controllo futuri.
//!
//! Server -> client (sempre JSON di testo):
//!   - `{"type":"ready"}`                       all'apertura
//!   - `{"type":"speech_start"}`                quando il VAD rileva voce
//!   - `{"type":"utterance","text":"..."}`      a fine enunciato trascritto
//!   - `{"type":"error","message":"..."}`       in caso di errore
//!
//! ## Protocollo `/tts/stream`
//!
//! Client -> server (JSON di testo):
//!   - `{"type":"config","voice":"if_sara","lang":"it","speed":1.0}`  (opzionale)
//!   - `{"type":"text","chunk":"..."}`   testo incrementale da Hermes
//!   - `{"type":"flush"}`                sintetizza subito il testo bufferizzato
//!   - `{"type":"end"}`                  flush finale e chiusura
//!
//! Server -> client:
//!   - `{"type":"ready"}`                                  all'apertura
//!   - `{"type":"audio_start","sample_rate":24000}`         prima di un blocco
//!   - messaggi **binari**: PCM f32 LE mono della frase sintetizzata
//!   - `{"type":"audio_end"}`                               fine del blocco
//!   - `{"type":"error","message":"..."}`                   in caso di errore

use std::sync::{Arc, Mutex};

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
};
use log::{debug, error, info, warn};
use serde_json::{json, Value};

use crate::api::ApiState;
use crate::audio_toolkit::constants::WHISPER_SAMPLE_RATE;
use crate::audio_toolkit::vad::{SmoothedVad, VadFrame};
use crate::audio_toolkit::{SileroVad, VoiceActivityDetector};
use crate::tts::KokoroTts;

/// Frame VAD da 30 ms a 16 kHz = 480 campioni (come il resto di Handy).
const VAD_FRAME_SAMPLES: usize = (WHISPER_SAMPLE_RATE as usize * 30) / 1000;

/// Eventi prodotti dall'analisi VAD di un blocco di campioni.
/// Disaccoppia l'analisi (che tiene un borrow su VAD/buffer) dall'I/O async.
enum SttEvent {
    SpeechStart,
    EndOfUtterance(Vec<f32>),
}

/// Handler Axum: fa l'upgrade della richiesta a WebSocket.
pub async fn stt_websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<ApiState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_stt_socket(socket, state))
}

async fn handle_stt_socket(mut socket: WebSocket, state: Arc<ApiState>) {
    info!("STT WS: connessione aperta");

    // VAD: Silero + smoothing, stessi parametri usati dal recorder di Handy.
    let vad_path = match state.vad_model_path.to_str() {
        Some(p) => p.to_string(),
        None => {
            error!("STT WS: percorso modello VAD non valido (UTF-8)");
            return;
        }
    };
    let mut vad: SmoothedVad = match SileroVad::new(&vad_path, 0.3) {
        Ok(silero) => SmoothedVad::new(Box::new(silero), 15, 15, 2),
        Err(e) => {
            error!("STT WS: inizializzazione VAD fallita: {}", e);
            let _ = socket
                .send(Message::Text(
                    json!({"type": "error", "message": "VAD init failed"}).to_string(),
                ))
                .await;
            return;
        }
    };

    let _ = socket
        .send(Message::Text(json!({"type": "ready"}).to_string()))
        .await;

    // Campioni ricevuti ma non ancora suddivisi in frame VAD.
    let mut sample_buf: Vec<f32> = Vec::with_capacity(VAD_FRAME_SAMPLES * 8);
    // Campioni dell'enunciato in corso (accumula i frame Speech del VAD).
    let mut utterance: Vec<f32> = Vec::new();
    let mut in_speech = false;

    while let Some(msg) = socket.recv().await {
        let msg = match msg {
            Ok(m) => m,
            Err(e) => {
                warn!("STT WS: errore di ricezione: {}", e);
                break;
            }
        };
        // Attivita' in corso: rinvia l'auto-unload dei modelli.
        state.call_models.touch();

        match msg {
            Message::Binary(bytes) => {
                if bytes.len() % 4 != 0 {
                    warn!("STT WS: payload binario non multiplo di 4 byte, scartato");
                    continue;
                }
                sample_buf.extend(
                    bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
                );

                // Analisi VAD: raccoglie eventi senza fare I/O (niente borrow
                // tenuto attraverso un await).
                let events = analyze_frames(&mut vad, &mut sample_buf, &mut utterance, &mut in_speech);

                // Gestione eventi: qui si puo' fare I/O async liberamente.
                for event in events {
                    match event {
                        SttEvent::SpeechStart => {
                            let _ = socket
                                .send(Message::Text(
                                    json!({"type": "speech_start"}).to_string(),
                                ))
                                .await;
                        }
                        SttEvent::EndOfUtterance(audio) => {
                            finalize_utterance(&mut socket, &state, audio).await;
                        }
                    }
                }
            }
            Message::Text(txt) => {
                debug!("STT WS: messaggio di controllo ignorato: {}", txt);
            }
            Message::Close(_) => {
                info!("STT WS: chiusura richiesta dal client");
                break;
            }
            Message::Ping(_) | Message::Pong(_) => {}
        }
    }

    // In coda: se resta un enunciato non finalizzato, trascrivilo comunque.
    if !utterance.is_empty() {
        finalize_utterance(&mut socket, &state, std::mem::take(&mut utterance)).await;
    }
    info!("STT WS: connessione chiusa");
}

/// Estrae i frame VAD completi da `sample_buf`, aggiorna lo stato di voce e
/// accumula l'enunciato. Ritorna gli eventi da gestire (con I/O) dopo.
fn analyze_frames(
    vad: &mut SmoothedVad,
    sample_buf: &mut Vec<f32>,
    utterance: &mut Vec<f32>,
    in_speech: &mut bool,
) -> Vec<SttEvent> {
    let mut events = Vec::new();
    let mut offset = 0;

    while offset + VAD_FRAME_SAMPLES <= sample_buf.len() {
        let frame = &sample_buf[offset..offset + VAD_FRAME_SAMPLES];
        match vad.push_frame(frame) {
            Ok(VadFrame::Speech(speech)) => {
                if !*in_speech {
                    *in_speech = true;
                    events.push(SttEvent::SpeechStart);
                }
                utterance.extend_from_slice(speech);
            }
            Ok(VadFrame::Noise) => {
                // Transizione voce -> silenzio: l'hangover del VAD garantisce
                // che sia gia' passato abbastanza silenzio. Fine enunciato.
                if *in_speech && !utterance.is_empty() {
                    *in_speech = false;
                    events.push(SttEvent::EndOfUtterance(std::mem::take(utterance)));
                }
            }
            Err(e) => warn!("STT WS: errore VAD su un frame: {}", e),
        }
        offset += VAD_FRAME_SAMPLES;
    }

    sample_buf.drain(0..offset);
    events
}

/// Trascrive un enunciato completo (operazione bloccante) e invia il testo.
async fn finalize_utterance(socket: &mut WebSocket, state: &Arc<ApiState>, audio: Vec<f32>) {
    let sample_count = audio.len();
    debug!("STT WS: trascrizione enunciato ({} campioni)", sample_count);

    let tm = state.transcription_manager.clone();
    let result = tokio::task::spawn_blocking(move || {
        tm.initiate_model_load();
        tm.transcribe(audio)
    })
    .await;

    let payload = match result {
        Ok(Ok(text)) => {
            let text = text.trim();
            if text.is_empty() {
                return; // niente di trascritto: non inviare nulla
            }
            info!("STT WS: enunciato -> {}", text);
            json!({"type": "utterance", "text": text, "samples": sample_count})
        }
        Ok(Err(e)) => {
            error!("STT WS: trascrizione fallita: {}", e);
            json!({"type": "error", "message": format!("transcription failed: {e}")})
        }
        Err(e) => {
            error!("STT WS: task di trascrizione interrotto: {}", e);
            json!({"type": "error", "message": "transcription task panicked"})
        }
    };

    let _ = socket.send(Message::Text(payload.to_string())).await;
}

// ============================================================================
//  TTS in streaming  —  /tts/stream
// ============================================================================

/// Se il testo bufferizzato supera questa lunghezza senza punteggiatura di
/// fine frase, lo sintetizziamo comunque (evita latenza illimitata in attesa
/// di un punto che potrebbe non arrivare).
const TTS_MAX_BUFFER_CHARS: usize = 200;

/// Handler Axum: fa l'upgrade della richiesta a WebSocket per lo streaming TTS.
pub async fn tts_websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<ApiState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_tts_socket(socket, state))
}

async fn handle_tts_socket(mut socket: WebSocket, state: Arc<ApiState>) {
    info!("TTS WS: connessione aperta");

    // Usa l'engine TTS gia' caricato in VRAM dal canale di controllo
    // (POST /models/load). Se non e' caricato, l'orchestrator deve caricarlo
    // prima di aprire questo WebSocket.
    let tts = match state.call_models.tts_engine() {
        Some(engine) => engine,
        None => {
            error!("TTS WS: motore Kokoro non caricato");
            let _ = socket
                .send(Message::Text(
                    json!({
                        "type": "error",
                        "message": "TTS model not loaded: call POST /models/load first"
                    })
                    .to_string(),
                ))
                .await;
            return;
        }
    };
    state.call_models.touch();

    // Parametri di sintesi, sovrascrivibili dal messaggio "config".
    let mut voice = "if_sara".to_string();
    let mut lang = "it".to_string();
    let mut speed = 1.0f32;
    // Testo ricevuto da Hermes ma non ancora sintetizzato.
    let mut text_buf = String::new();

    let _ = socket
        .send(Message::Text(json!({"type": "ready"}).to_string()))
        .await;

    while let Some(msg) = socket.recv().await {
        let msg = match msg {
            Ok(m) => m,
            Err(e) => {
                warn!("TTS WS: errore di ricezione: {}", e);
                break;
            }
        };
        // Attivita' in corso: rinvia l'auto-unload dei modelli.
        state.call_models.touch();

        match msg {
            Message::Text(txt) => {
                let parsed: Value = match serde_json::from_str(&txt) {
                    Ok(v) => v,
                    Err(_) => {
                        warn!("TTS WS: JSON non valido: {}", txt);
                        continue;
                    }
                };
                match parsed.get("type").and_then(Value::as_str) {
                    Some("config") => {
                        if let Some(v) = parsed.get("voice").and_then(Value::as_str) {
                            voice = v.to_string();
                        }
                        if let Some(l) = parsed.get("lang").and_then(Value::as_str) {
                            lang = l.to_string();
                        }
                        if let Some(s) = parsed.get("speed").and_then(Value::as_f64) {
                            speed = s as f32;
                        }
                        debug!(
                            "TTS WS: config voice={} lang={} speed={}",
                            voice, lang, speed
                        );
                    }
                    Some("text") => {
                        if let Some(chunk) = parsed.get("chunk").and_then(Value::as_str) {
                            text_buf.push_str(chunk);
                            // Sintetizza tutte le frasi gia' complete.
                            while let Some(sentence) = take_sentence(&mut text_buf) {
                                synthesize_and_send(
                                    &mut socket, &tts, &sentence, &voice, &lang, speed,
                                )
                                .await;
                            }
                        }
                    }
                    Some("flush") => {
                        let pending = std::mem::take(&mut text_buf);
                        let pending = pending.trim();
                        if !pending.is_empty() {
                            synthesize_and_send(
                                &mut socket, &tts, pending, &voice, &lang, speed,
                            )
                            .await;
                        }
                    }
                    Some("end") => {
                        let pending = std::mem::take(&mut text_buf);
                        let pending = pending.trim();
                        if !pending.is_empty() {
                            synthesize_and_send(
                                &mut socket, &tts, pending, &voice, &lang, speed,
                            )
                            .await;
                        }
                        info!("TTS WS: 'end' ricevuto, chiudo");
                        break;
                    }
                    other => debug!("TTS WS: tipo messaggio non gestito: {:?}", other),
                }
            }
            Message::Close(_) => {
                info!("TTS WS: chiusura richiesta dal client");
                break;
            }
            Message::Binary(_) => warn!("TTS WS: binario inatteso dal client, ignorato"),
            Message::Ping(_) | Message::Pong(_) => {}
        }
    }

    // L'engine TTS resta caricato in VRAM (gestito da CallModelManager): si
    // scarichera' da solo dopo 15 min di inattivita' o su /models/unload.
    info!("TTS WS: connessione chiusa");
}

/// Estrae dalla testa di `buf` la prima frase completa (fino a `.`, `!`, `?`
/// o newline). Se `buf` supera [`TTS_MAX_BUFFER_CHARS`] senza un confine, lo
/// restituisce tutto. Ritorna `None` se non c'e' ancora una frase completa.
fn take_sentence(buf: &mut String) -> Option<String> {
    loop {
        let boundary = buf.char_indices().find_map(|(i, c)| {
            matches!(c, '.' | '!' | '?' | '\n').then(|| i + c.len_utf8())
        });
        let cut = match boundary {
            Some(b) => b,
            None if buf.chars().count() >= TTS_MAX_BUFFER_CHARS => buf.len(),
            None => return None,
        };
        let sentence: String = buf.drain(..cut).collect();
        let trimmed = sentence.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
        // Era solo punteggiatura o spazi: riprova col resto del buffer.
        if buf.is_empty() {
            return None;
        }
    }
}

/// Sintetizza una frase con Kokoro (operazione bloccante) e invia l'audio.
async fn synthesize_and_send(
    socket: &mut WebSocket,
    tts: &Arc<Mutex<KokoroTts>>,
    text: &str,
    voice: &str,
    lang: &str,
    speed: f32,
) {
    debug!("TTS WS: sintesi -> {}", text);

    let tts = tts.clone();
    let (text, voice, lang) = (text.to_string(), voice.to_string(), lang.to_string());
    let result = tokio::task::spawn_blocking(move || {
        let mut engine = tts.lock().expect("mutex KokoroTts avvelenato");
        engine.synthesize(&text, &voice, &lang, speed)
    })
    .await;

    match result {
        Ok(Ok(audio)) => {
            let _ = socket
                .send(Message::Text(
                    json!({"type": "audio_start", "sample_rate": audio.sample_rate})
                        .to_string(),
                ))
                .await;
            let mut bytes = Vec::with_capacity(audio.samples.len() * 4);
            for sample in &audio.samples {
                bytes.extend_from_slice(&sample.to_le_bytes());
            }
            let _ = socket.send(Message::Binary(bytes)).await;
            let _ = socket
                .send(Message::Text(json!({"type": "audio_end"}).to_string()))
                .await;
        }
        Ok(Err(e)) => {
            error!("TTS WS: sintesi fallita: {}", e);
            let _ = socket
                .send(Message::Text(
                    json!({"type": "error", "message": format!("synthesis failed: {e}")})
                        .to_string(),
                ))
                .await;
        }
        Err(e) => {
            error!("TTS WS: task di sintesi interrotto: {}", e);
            let _ = socket
                .send(Message::Text(
                    json!({"type": "error", "message": "synthesis task panicked"}).to_string(),
                ))
                .await;
        }
    }
}

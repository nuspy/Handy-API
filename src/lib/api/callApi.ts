// Client HTTP verso l'API REST embedded di Handy (axum, default :8720).
// Usata dalla GUI per la gestione delle voci clonate e l'anteprima TTS;
// le stesse route servono anche integrazioni esterne (call-bridge, Control
// Center di callAPIcall), quindi i tipi rispecchiano i payload JSON 1:1.

const API_BASE = "http://127.0.0.1:8720";

export interface ClonedVoice {
  name: string;
  format: string;
  size_bytes: number;
  modified: string;
}

export interface TtsTestResult {
  status: string;
  audio_b64: string;
  sample_rate: number;
  duration_ms: number;
  voice: string;
  message?: string;
}

async function jsonOrThrow(res: Response): Promise<any> {
  const body = await res.json().catch(() => ({}));
  if (!res.ok && body?.message === undefined) {
    throw new Error(`HTTP ${res.status}`);
  }
  return body;
}

/** Elenco delle voci clonate (GET /voices/cloned). */
export async function listClonedVoices(): Promise<ClonedVoice[]> {
  const res = await fetch(`${API_BASE}/voices/cloned`);
  const body = await jsonOrThrow(res);
  return body.voices ?? [];
}

/** Converte un Blob audio in base64 (senza il prefisso data:). */
async function blobToBase64(blob: Blob): Promise<string> {
  const buf = new Uint8Array(await blob.arrayBuffer());
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < buf.length; i += chunk) {
    binary += String.fromCharCode(...buf.subarray(i, i + chunk));
  }
  return btoa(binary);
}

/** Salva un campione voce per il cloning (POST /voices/clone). */
export async function cloneVoice(
  name: string,
  audio: Blob,
  format: "wav" | "mp3" | "flac" | "ogg",
): Promise<string[]> {
  const audio_b64 = await blobToBase64(audio);
  const res = await fetch(`${API_BASE}/voices/clone`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, audio_b64, format }),
  });
  const body = await jsonOrThrow(res);
  if (body.status !== "ok") {
    throw new Error(body.message ?? "clonazione fallita");
  }
  return body.voices ?? [];
}

/** Elimina una voce clonata (DELETE /voices/cloned/:name). */
export async function deleteClonedVoice(name: string): Promise<string[]> {
  const res = await fetch(
    `${API_BASE}/voices/cloned/${encodeURIComponent(name)}`,
    { method: "DELETE" },
  );
  const body = await jsonOrThrow(res);
  if (body.status !== "ok") {
    throw new Error(body.message ?? "cancellazione fallita");
  }
  return body.voices ?? [];
}

/**
 * Sintesi one-shot per provare una voce (POST /tts/test).
 * Carica l'engine TTS se serve: la prima chiamata puo' richiedere decine di
 * secondi (Chatterbox scalda i pesi), per questo niente timeout breve.
 */
export async function ttsTest(
  text: string,
  opts: { voice?: string; lang?: string; speed?: number } = {},
): Promise<TtsTestResult> {
  const res = await fetch(`${API_BASE}/tts/test`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, ...opts }),
  });
  const body = (await jsonOrThrow(res)) as TtsTestResult;
  if (body.status !== "ok") {
    throw new Error(body.message ?? "sintesi fallita");
  }
  return body;
}

/** URL blob riproducibile da un WAV base64 (ricordarsi revokeObjectURL). */
export function wavUrlFromBase64(b64: string): string {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
}

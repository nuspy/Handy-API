import React, { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Circle, Square } from "lucide-react";
import { Button } from "../../ui/Button";
import { encodeWav } from "../../../lib/utils/wavEncode";

/** Durata minima/massima del campione per un cloning di qualita'. */
const MIN_SECONDS = 5;
const MAX_SECONDS = 15;

interface VoiceRecorderProps {
  /** Chiamata a registrazione completata (WAV mono 16 bit). */
  onRecorded: (wav: Blob) => void;
  disabled?: boolean;
}

/**
 * Registratore del campione voce: cattura PCM via Web Audio (MediaRecorder
 * produrrebbe webm/opus, che /voices/clone non accetta), indicatore di
 * livello via AnalyserNode, stop abilitato dopo MIN_SECONDS e automatico a
 * MAX_SECONDS.
 */
export const VoiceRecorder: React.FC<VoiceRecorderProps> = ({
  onRecorded,
  disabled = false,
}) => {
  const { t } = useTranslation();
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [level, setLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const startedAtRef = useRef(0);
  const rafRef = useRef<number>();
  const recordingRef = useRef(false);

  const cleanup = () => {
    recordingRef.current = false;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = undefined;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    ctxRef.current?.close().catch(() => {});
    ctxRef.current = null;
  };

  useEffect(() => cleanup, []);

  const start = async () => {
    setError(null);
    chunksRef.current = [];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false },
      });
      streamRef.current = stream;
      const ctx = new AudioContext();
      ctxRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);

      // ScriptProcessor e' deprecato ma resta il modo piu' semplice e
      // universale in WebView2 per avere i campioni raw senza AudioWorklet.
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      source.connect(processor);
      processor.connect(ctx.destination);
      processor.onaudioprocess = (e) => {
        if (!recordingRef.current) return;
        chunksRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      };

      recordingRef.current = true;
      startedAtRef.current = performance.now();
      setRecording(true);
      setElapsed(0);

      const buf = new Float32Array(analyser.fftSize);
      const tick = () => {
        if (!recordingRef.current) return;
        const secs = (performance.now() - startedAtRef.current) / 1000;
        setElapsed(secs);
        analyser.getFloatTimeDomainData(buf);
        let sum = 0;
        for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
        setLevel(Math.min(1, Math.sqrt(sum / buf.length) * 4));
        if (secs >= MAX_SECONDS) {
          stop();
          return;
        }
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    } catch (e) {
      cleanup();
      setError(
        e instanceof Error ? e.message : t("settings.voices.recorder.micError"),
      );
    }
  };

  const stop = () => {
    const ctx = ctxRef.current;
    const sampleRate = ctx?.sampleRate ?? 48000;
    cleanup();
    setRecording(false);
    setLevel(0);

    const total = chunksRef.current.reduce((n, c) => n + c.length, 0);
    if (total / sampleRate < MIN_SECONDS) {
      setError(t("settings.voices.recorder.tooShort", { min: MIN_SECONDS }));
      return;
    }
    const samples = new Float32Array(total);
    let offset = 0;
    for (const c of chunksRef.current) {
      samples.set(c, offset);
      offset += c.length;
    }
    chunksRef.current = [];
    onRecorded(encodeWav(samples, sampleRate));
  };

  const canStop = elapsed >= MIN_SECONDS;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3">
        {!recording ? (
          <Button
            variant="primary-soft"
            size="sm"
            onClick={start}
            disabled={disabled}
          >
            <span className="flex items-center gap-1.5">
              <Circle width={12} height={12} fill="currentColor" className="text-red-500" />
              {t("settings.voices.recorder.start")}
            </span>
          </Button>
        ) : (
          <Button
            variant="danger"
            size="sm"
            onClick={stop}
            disabled={!canStop}
            title={
              canStop
                ? undefined
                : t("settings.voices.recorder.waitMin", { min: MIN_SECONDS })
            }
          >
            <span className="flex items-center gap-1.5">
              <Square width={12} height={12} fill="currentColor" />
              {t("settings.voices.recorder.stop")}
            </span>
          </Button>
        )}
        {recording && (
          <span className="text-sm tabular-nums text-text/70">
            {elapsed.toFixed(1)}s / {MAX_SECONDS}s
          </span>
        )}
      </div>

      {recording && (
        <div className="h-2 w-full rounded bg-mid-gray/20 overflow-hidden">
          <div
            className="h-full bg-logo-primary transition-[width] duration-75"
            style={{ width: `${Math.round(level * 100)}%` }}
          />
        </div>
      )}

      <p className="text-xs text-text/50">
        {t("settings.voices.recorder.hint", { min: MIN_SECONDS, max: MAX_SECONDS })}
      </p>

      {error && <p className="text-xs text-red-400">{error}</p>}
    </div>
  );
};

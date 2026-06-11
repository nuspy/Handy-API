import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { ask } from "@tauri-apps/plugin-dialog";
import { Trash2 } from "lucide-react";
import Badge from "../../ui/Badge";
import { Button } from "../../ui/Button";
import { AudioPlayer } from "../../ui/AudioPlayer";
import {
  type ClonedVoice,
  deleteClonedVoice,
  ttsTest,
  wavUrlFromBase64,
} from "../../../lib/api/callApi";

interface ClonedVoiceCardProps {
  voice: ClonedVoice;
  /** True se questa e' la voce preferita nei settings (selected_tts_voice). */
  isActive: boolean;
  /** Testo di prova condiviso dalla sezione (modificabile dall'utente). */
  sampleText: string;
  /** Imposta/azzera la voce preferita. */
  onSetActive: (name: string | null) => Promise<void>;
  /** Notifica la sezione che la voce e' stata eliminata. */
  onDeleted: () => void;
}

function formatSize(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

export const ClonedVoiceCard: React.FC<ClonedVoiceCardProps> = ({
  voice,
  isActive,
  sampleText,
  onSetActive,
  onDeleted,
}) => {
  const { t } = useTranslation();
  const [busy, setBusy] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);

  // L'AudioPlayer carica l'anteprima in modo lazy: la sintesi parte solo al
  // primo play (la prima volta puo' scaldare l'engine Chatterbox, decine di s).
  const loadPreview = async (): Promise<string | null> => {
    setPreviewError(null);
    try {
      const res = await ttsTest(sampleText, { voice: voice.name });
      return wavUrlFromBase64(res.audio_b64);
    } catch (e) {
      setPreviewError(
        e instanceof Error ? e.message : t("settings.voices.card.previewError"),
      );
      return null;
    }
  };

  const handleDelete = async () => {
    const confirmed = await ask(
      t("settings.voices.card.deleteConfirm", { name: voice.name }),
      { title: t("settings.voices.card.deleteTitle"), kind: "warning" },
    );
    if (!confirmed) return;
    setBusy(true);
    try {
      await deleteClonedVoice(voice.name);
      if (isActive) await onSetActive(null);
      onDeleted();
    } catch (e) {
      console.error(`Eliminazione voce ${voice.name} fallita:`, e);
    } finally {
      setBusy(false);
    }
  };

  const handleSetActive = async () => {
    setBusy(true);
    try {
      await onSetActive(isActive ? null : voice.name);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className={`rounded-lg border p-3 space-y-2 ${
        isActive ? "border-logo-primary/60 bg-logo-primary/5" : "border-mid-gray/20"
      }`}
    >
      <div className="flex items-center gap-2">
        <span className="font-medium text-sm truncate">{voice.name}</span>
        <Badge variant="secondary">{voice.format}</Badge>
        {isActive && (
          <Badge variant="success">{t("settings.voices.card.active")}</Badge>
        )}
        <span className="text-xs text-text/40 ms-auto tabular-nums">
          {formatSize(voice.size_bytes)}
        </span>
      </div>

      <div className="flex items-center gap-2">
        <AudioPlayer className="flex-1" onLoadRequest={loadPreview} />
        <Button
          variant={isActive ? "secondary" : "primary-soft"}
          size="sm"
          onClick={handleSetActive}
          disabled={busy}
        >
          {isActive
            ? t("settings.voices.card.unsetActive")
            : t("settings.voices.card.setActive")}
        </Button>
        <Button
          variant="danger-ghost"
          size="sm"
          onClick={handleDelete}
          disabled={busy}
          aria-label={t("settings.voices.card.deleteTitle")}
        >
          <Trash2 width={14} height={14} />
        </Button>
      </div>

      {previewError && <p className="text-xs text-red-400">{previewError}</p>}
    </div>
  );
};

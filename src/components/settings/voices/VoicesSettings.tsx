import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { AlertTriangle, Mic, Upload as UploadIcon } from "lucide-react";
import { Input } from "../../ui/Input";
import { Textarea } from "../../ui/Textarea";
import { useModelStore } from "@/stores/modelStore";
import { useSettingsStore } from "@/stores/settingsStore";
import type { ModelInfo } from "@/bindings";
import { type ClonedVoice, cloneVoice, listClonedVoices } from "../../../lib/api/callApi";
import { ClonedVoiceCard } from "./ClonedVoiceCard";
import { VoiceRecorder } from "./VoiceRecorder";
import { VoiceUpload, type VoiceFormat } from "./VoiceUpload";

type CreateMode = "record" | "upload";

const NAME_RE = /^[a-zA-Z0-9_-]+$/;

/**
 * Sezione "Voci": gestione dei campioni per il voice cloning Chatterbox.
 * Lista voci clonate (anteprima/attiva/elimina) + creazione nuova voce
 * registrando dal microfono o caricando un file audio.
 */
export const VoicesSettings: React.FC = () => {
  const { t } = useTranslation();
  const [voices, setVoices] = useState<ClonedVoice[]>([]);
  const [apiError, setApiError] = useState<string | null>(null);
  const [newName, setNewName] = useState("");
  const [createMode, setCreateMode] = useState<CreateMode>("record");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [sampleText, setSampleText] = useState("");

  const models = useModelStore((s) => s.models);
  const selectedTtsModel = useSettingsStore(
    (s) => s.settings?.selected_tts_model ?? "",
  );
  const selectedTtsVoice = useSettingsStore(
    (s) => s.settings?.selected_tts_voice ?? "",
  );
  const updateSetting = useSettingsStore((s) => s.updateSetting);

  const chatterboxModel = useMemo(
    () => models.find((m: ModelInfo) => m.engine_type === "Chatterbox"),
    [models],
  );
  const chatterboxReady = chatterboxModel?.is_downloaded ?? false;
  const chatterboxActive =
    chatterboxReady && selectedTtsModel === chatterboxModel?.id;

  const refresh = useCallback(async () => {
    try {
      setVoices(await listClonedVoices());
      setApiError(null);
    } catch (e) {
      setApiError(
        e instanceof Error ? e.message : t("settings.voices.apiError"),
      );
    }
  }, [t]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (!sampleText) setSampleText(t("settings.voices.sampleTextDefault"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const nameValid = NAME_RE.test(newName);

  const saveVoice = async (blob: Blob, format: VoiceFormat) => {
    if (!nameValid) {
      setSaveError(t("settings.voices.create.nameInvalid"));
      return;
    }
    setSaving(true);
    setSaveError(null);
    try {
      await cloneVoice(newName, blob, format);
      setNewName("");
      await refresh();
    } catch (e) {
      setSaveError(
        e instanceof Error ? e.message : t("settings.voices.create.failed"),
      );
    } finally {
      setSaving(false);
    }
  };

  // Imposta la voce preferita; se serve attiva anche l'engine Chatterbox
  // (stessa logica del tab TTS in ModelsSettings).
  const setActiveVoice = async (name: string | null) => {
    await updateSetting("selected_tts_voice", name ?? "");
    if (name && chatterboxReady && !chatterboxActive && chatterboxModel) {
      await updateSetting("selected_tts_model", chatterboxModel.id);
    }
  };

  return (
    <div className="max-w-3xl w-full mx-auto space-y-6">
      <div>
        <h1 className="text-xl font-semibold mb-2">
          {t("settings.voices.title")}
        </h1>
        <p className="text-sm text-text/60">
          {t("settings.voices.description")}
        </p>
      </div>

      {!chatterboxReady && (
        <div className="flex items-start gap-2 rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-300">
          <AlertTriangle width={16} height={16} className="mt-0.5 shrink-0" />
          <span>{t("settings.voices.chatterboxMissing")}</span>
        </div>
      )}

      {/* Creazione nuova voce */}
      <div className="rounded-lg border border-mid-gray/20 p-4 space-y-3">
        <h2 className="text-sm font-medium text-text/60">
          {t("settings.voices.create.title")}
        </h2>
        <Input
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          placeholder={t("settings.voices.create.namePlaceholder")}
          disabled={saving}
        />
        {newName && !nameValid && (
          <p className="text-xs text-red-400">
            {t("settings.voices.create.nameInvalid")}
          </p>
        )}

        <div className="flex gap-2 border-b border-mid-gray/30">
          {(
            [
              ["record", Mic, t("settings.voices.create.tabRecord")],
              ["upload", UploadIcon, t("settings.voices.create.tabUpload")],
            ] as const
          ).map(([mode, Icon, label]) => (
            <button
              key={mode}
              type="button"
              onClick={() => setCreateMode(mode)}
              className={`flex items-center gap-1.5 px-3 py-2 text-sm font-medium border-b-2 -mb-px transition-colors ${
                createMode === mode
                  ? "border-logo-primary text-logo-primary"
                  : "border-transparent text-text/60 hover:text-text"
              }`}
            >
              <Icon width={14} height={14} />
              {label}
            </button>
          ))}
        </div>

        {createMode === "record" ? (
          <VoiceRecorder
            disabled={saving || !nameValid}
            onRecorded={(wav) => saveVoice(wav, "wav")}
          />
        ) : (
          <VoiceUpload
            disabled={saving || !nameValid}
            onSelected={(file, format) => saveVoice(file, format)}
          />
        )}

        {saving && (
          <p className="text-xs text-text/50">
            {t("settings.voices.create.saving")}
          </p>
        )}
        {saveError && <p className="text-xs text-red-400">{saveError}</p>}
      </div>

      {/* Testo di prova per le anteprime */}
      <div className="space-y-1">
        <h2 className="text-sm font-medium text-text/60">
          {t("settings.voices.sampleTextLabel")}
        </h2>
        <Textarea
          value={sampleText}
          onChange={(e) => setSampleText(e.target.value)}
          rows={2}
        />
      </div>

      {/* Lista voci clonate */}
      <div className="space-y-3">
        <h2 className="text-sm font-medium text-text/60">
          {t("settings.voices.listTitle")}
        </h2>
        {apiError ? (
          <p className="text-sm text-red-400">{apiError}</p>
        ) : voices.length === 0 ? (
          <p className="text-sm text-text/50">{t("settings.voices.empty")}</p>
        ) : (
          voices.map((v) => (
            <ClonedVoiceCard
              key={v.name}
              voice={v}
              isActive={v.name === selectedTtsVoice}
              sampleText={sampleText}
              onSetActive={setActiveVoice}
              onDeleted={refresh}
            />
          ))
        )}
      </div>
    </div>
  );
};

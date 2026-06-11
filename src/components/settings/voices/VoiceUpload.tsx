import React, { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Upload } from "lucide-react";
import { Button } from "../../ui/Button";

export type VoiceFormat = "wav" | "mp3" | "flac" | "ogg";

const ACCEPTED: VoiceFormat[] = ["wav", "mp3", "flac", "ogg"];
const MAX_BYTES = 25 * 1024 * 1024;

interface VoiceUploadProps {
  /** Chiamata con il file scelto e il suo formato (dall'estensione). */
  onSelected: (file: File, format: VoiceFormat) => void;
  disabled?: boolean;
}

/** Upload di un campione voce esistente (wav/mp3/flac/ogg, max 25 MB). */
export const VoiceUpload: React.FC<VoiceUploadProps> = ({
  onSelected,
  disabled = false,
}) => {
  const { t } = useTranslation();
  const inputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    const file = e.target.files?.[0];
    e.target.value = ""; // permetti di riselezionare lo stesso file
    if (!file) return;
    const ext = file.name.split(".").pop()?.toLowerCase() as VoiceFormat;
    if (!ACCEPTED.includes(ext)) {
      setError(t("settings.voices.upload.badFormat"));
      return;
    }
    if (file.size > MAX_BYTES) {
      setError(t("settings.voices.upload.tooBig"));
      return;
    }
    onSelected(file, ext);
  };

  return (
    <div className="space-y-2">
      <input
        ref={inputRef}
        type="file"
        accept=".wav,.mp3,.flac,.ogg"
        className="hidden"
        onChange={handleChange}
      />
      <Button
        variant="secondary"
        size="sm"
        onClick={() => inputRef.current?.click()}
        disabled={disabled}
      >
        <span className="flex items-center gap-1.5">
          <Upload width={14} height={14} />
          {t("settings.voices.upload.choose")}
        </span>
      </Button>
      <p className="text-xs text-text/50">{t("settings.voices.upload.hint")}</p>
      {error && <p className="text-xs text-red-400">{error}</p>}
    </div>
  );
};

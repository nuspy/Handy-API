package com.handy.voice.tts

import android.content.Context
import com.handy.voice.models.ModelManager
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Qwen3-TTS engine using llama.cpp for audio token generation
 * and ONNX Runtime for audio token decoding.
 *
 * Pipeline: Text -> llama.cpp (audio tokens) -> ONNX decoder (waveform)
 */
@Singleton
class QwenTTSEngine @Inject constructor(
    @ApplicationContext private val context: Context,
    private val modelManager: ModelManager,
    private val audioTokenDecoder: AudioTokenDecoder,
) {
    private var statePtr: Long = 0L
    private val lock = Any()

    companion object {
        init {
            System.loadLibrary("handyvoice")
        }

        // Language codes supported by Qwen3-TTS
        val SUPPORTED_LANGUAGES = mapOf(
            "auto" to "auto",
            "en" to "English",
            "zh" to "Chinese",
            "ja" to "Japanese",
            "ko" to "Korean",
            "de" to "German",
            "fr" to "French",
            "ru" to "Russian",
            "pt" to "Portuguese",
            "es" to "Spanish",
            "it" to "Italian",
        )
    }

    fun initialize(modelPath: String? = null): Boolean {
        synchronized(lock) {
            release()
            val path = modelPath ?: modelManager.getModelPath("tts") ?: return false
            statePtr = nativeInit(path, 2048, 4)
            return statePtr != 0L
        }
    }

    /**
     * Synthesize speech from text.
     *
     * @param text The text to speak
     * @param language Language code (e.g., "en", "it", "zh")
     * @param voiceProfile Optional voice profile ID for voice cloning
     * @param onProgress Progress callback (0.0 to 1.0)
     * @return PCM float audio data at 24kHz, or null on failure
     */
    fun synthesize(
        text: String,
        language: String = "en",
        voiceProfile: String? = null,
        onProgress: ((Float) -> Unit)? = null,
    ): FloatArray? {
        synchronized(lock) {
            if (statePtr == 0L) {
                if (!initialize()) return null
            }

            onProgress?.invoke(0.1f)

            // Build the TTS prompt
            val prompt = buildTTSPrompt(text, language, voiceProfile)

            onProgress?.invoke(0.2f)

            // Generate audio tokens using llama.cpp
            val audioTokens = nativeGenerate(statePtr, prompt, 2048, 0.7f)
            if (audioTokens.isEmpty()) return null

            onProgress?.invoke(0.6f)

            // Decode audio tokens to waveform using ONNX decoder
            val waveform = audioTokenDecoder.decode(audioTokens)

            onProgress?.invoke(1.0f)
            return waveform
        }
    }

    /**
     * Build the prompt for Qwen3-TTS inference.
     *
     * The exact format depends on the model variant:
     * - Base model: supports voice cloning with reference audio
     * - CustomVoice model: supports preset speakers with instructions
     *
     * NOTE: The prompt format may need adjustment based on the verification
     * script results (verify_qwen3_tts.py). The format below is based on
     * the official documentation and may need to be updated.
     */
    private fun buildTTSPrompt(
        text: String,
        language: String,
        voiceProfile: String?,
    ): String {
        val langName = SUPPORTED_LANGUAGES[language] ?: language

        return if (voiceProfile != null) {
            // Voice cloning mode: include reference audio path
            val profilePath = modelManager.getVoiceProfilePath(voiceProfile)
            "<|audio_bos|><|SPEECH_GENERATION_START|>" +
                "<|language:$langName|>" +
                "<|reference_audio:$profilePath|>" +
                "<|text:$text|>" +
                "<|SPEECH_GENERATION_END|>"
        } else {
            // Standard TTS mode
            "<|audio_bos|><|SPEECH_GENERATION_START|>" +
                "<|language:$langName|>" +
                "<|text:$text|>" +
                "<|SPEECH_GENERATION_END|>"
        }
    }

    fun release() {
        synchronized(lock) {
            if (statePtr != 0L) {
                nativeRelease(statePtr)
                statePtr = 0L
            }
        }
    }

    fun isLoaded(): Boolean = statePtr != 0L

    fun getSystemInfo(): String = nativeGetSystemInfo()

    // JNI methods
    private external fun nativeInit(modelPath: String, nCtx: Int, nThreads: Int): Long
    private external fun nativeRelease(statePtr: Long)
    private external fun nativeGenerate(
        statePtr: Long,
        prompt: String,
        maxTokens: Int,
        temperature: Float,
    ): IntArray
    private external fun nativeGenerateText(
        statePtr: Long,
        prompt: String,
        maxTokens: Int,
        temperature: Float,
    ): String
    private external fun nativeGetSystemInfo(): String
}

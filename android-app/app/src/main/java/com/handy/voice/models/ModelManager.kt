package com.handy.voice.models

import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Manages model files on disk and their lifecycle in memory.
 *
 * Models are stored in app-specific external storage under "models/".
 * Each model type has a known filename.
 */
@Singleton
class ModelManager @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    private val modelsDir: File
        get() = File(context.getExternalFilesDir(null), "models").also { it.mkdirs() }

    private val voiceProfilesDir: File
        get() = File(context.filesDir, "voice_profiles").also { it.mkdirs() }

    /** Known model filenames for each type */
    private val modelFiles = mapOf(
        "stt" to "whisper-small-q5_1.bin",
        "tts" to "qwen3-tts-0.6b-base-q4_k_m.gguf",
        "tts_decoder" to "qwen3-tts-tokenizer-12hz.onnx",
        "chat" to "qwen3-0.6b-q4_k_m.gguf",
    )

    fun isModelAvailable(modelType: String): Boolean {
        val filename = modelFiles[modelType] ?: return false
        return File(modelsDir, filename).exists()
    }

    fun getModelPath(modelType: String): String? {
        val filename = modelFiles[modelType] ?: return null
        val file = File(modelsDir, filename)
        return if (file.exists()) file.absolutePath else null
    }

    fun getModelFile(modelType: String): File? {
        val filename = modelFiles[modelType] ?: return null
        return File(modelsDir, filename)
    }

    fun getModelsDirectory(): File = modelsDir

    fun getVoiceProfilePath(profileId: String): String? {
        val file = File(voiceProfilesDir, "$profileId.wav")
        return if (file.exists()) file.absolutePath else null
    }

    /**
     * Get total size of all downloaded models.
     */
    fun getTotalModelsSize(): Long {
        return modelsDir.listFiles()?.sumOf { it.length() } ?: 0L
    }

    /**
     * Delete a model from disk.
     */
    fun deleteModel(modelType: String): Boolean {
        val filename = modelFiles[modelType] ?: return false
        return File(modelsDir, filename).delete()
    }
}

package com.handy.voice.tts

import android.content.Context
import com.handy.voice.models.ModelManager
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

/**
 * LLM engine for chat functionality. Reuses llama.cpp via QwenTTSEngine's JNI bridge.
 *
 * This can load a separate chat model (e.g., Qwen3-0.6B, Llama-3.2-1B)
 * independent from the TTS model.
 */
@Singleton
class LLMEngine @Inject constructor(
    @ApplicationContext private val context: Context,
    private val modelManager: ModelManager,
) {
    private var statePtr: Long = 0L
    private val lock = Any()

    companion object {
        init {
            System.loadLibrary("handyvoice")
        }
    }

    fun initialize(modelPath: String? = null): Boolean {
        synchronized(lock) {
            release()
            val path = modelPath ?: modelManager.getModelPath("chat") ?: return false
            statePtr = nativeInit(path, 4096, 4)
            return statePtr != 0L
        }
    }

    /**
     * Generate text response from prompt.
     */
    fun generate(
        prompt: String,
        maxTokens: Int = 512,
        temperature: Float = 0.7f,
    ): String {
        synchronized(lock) {
            if (statePtr == 0L) {
                if (!initialize()) {
                    return "[error: chat model not loaded. Download it in Settings.]"
                }
            }
            return nativeGenerateText(statePtr, prompt, maxTokens, temperature)
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

    // Reuses the same JNI as QwenTTSEngine (same native library)
    private external fun nativeInit(modelPath: String, nCtx: Int, nThreads: Int): Long
    private external fun nativeRelease(statePtr: Long)
    private external fun nativeGenerateText(
        statePtr: Long,
        prompt: String,
        maxTokens: Int,
        temperature: Float,
    ): String
}

package com.handy.voice.stt

import android.content.Context
import com.handy.voice.models.ModelManager
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Kotlin wrapper for whisper.cpp via JNI.
 *
 * Handles model loading, transcription, and lifecycle management.
 * Thread-safe: only one transcription can run at a time.
 */
@Singleton
class WhisperEngine @Inject constructor(
    @ApplicationContext private val context: Context,
    private val modelManager: ModelManager,
) {
    private var contextPtr: Long = 0L
    private val lock = Any()

    companion object {
        init {
            System.loadLibrary("handyvoice")
        }
    }

    /**
     * Initialize whisper with the given model. Call before transcribe().
     * If already initialized with a different model, the old model is released first.
     */
    fun initialize(modelPath: String? = null): Boolean {
        synchronized(lock) {
            release()
            val path = modelPath ?: modelManager.getModelPath("stt") ?: return false
            contextPtr = nativeInit(path)
            return contextPtr != 0L
        }
    }

    /**
     * Transcribe audio samples (16kHz mono float32, range [-1, 1]).
     */
    fun transcribe(
        audioData: FloatArray,
        language: String = "auto",
        translate: Boolean = false,
    ): String {
        synchronized(lock) {
            if (contextPtr == 0L) {
                if (!initialize()) {
                    return "[error: model not loaded]"
                }
            }
            return nativeTranscribe(contextPtr, audioData, language, translate)
        }
    }

    fun release() {
        synchronized(lock) {
            if (contextPtr != 0L) {
                nativeRelease(contextPtr)
                contextPtr = 0L
            }
        }
    }

    fun getSystemInfo(): String = nativeGetSystemInfo()

    fun isLoaded(): Boolean = contextPtr != 0L

    // JNI methods
    private external fun nativeInit(modelPath: String): Long
    private external fun nativeRelease(contextPtr: Long)
    private external fun nativeTranscribe(
        contextPtr: Long,
        audioData: FloatArray,
        language: String,
        translate: Boolean,
    ): String
    private external fun nativeGetSystemInfo(): String
}

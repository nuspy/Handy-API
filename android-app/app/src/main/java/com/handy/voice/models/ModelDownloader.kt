package com.handy.voice.models

import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Downloads model files from Hugging Face Hub with progress reporting and resume support.
 */
@Singleton
class ModelDownloader @Inject constructor(
    @ApplicationContext private val context: Context,
    private val modelManager: ModelManager,
) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(5, TimeUnit.MINUTES)
        .build()

    /** Download URLs for each model type (Hugging Face Hub) */
    private val modelUrls = mapOf(
        "stt" to "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin",
        "tts" to "https://huggingface.co/Qwen/Qwen3-TTS-0.6B-Base-GGUF/resolve/main/qwen3-tts-0.6b-base-q4_k_m.gguf",
        // NOTE: tts_decoder URL will be determined after running verify_qwen3_tts.py
        // For now, this is a placeholder
        "tts_decoder" to "https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz/resolve/main/tokenizer.onnx",
        "chat" to "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/qwen3-0.6b-q4_k_m.gguf",
    )

    /**
     * Download a model with progress reporting.
     *
     * Supports resume: if a partial download exists, it will continue from where it left off.
     *
     * @param modelType One of "stt", "tts", "tts_decoder", "chat"
     * @param onProgress Callback with progress value (0.0 to 1.0)
     */
    suspend fun download(
        modelType: String,
        onProgress: (Float) -> Unit,
    ) = withContext(Dispatchers.IO) {
        val url = modelUrls[modelType]
            ?: throw IllegalArgumentException("Unknown model type: $modelType")

        val targetFile = modelManager.getModelFile(modelType)
            ?: throw IllegalStateException("No file mapping for model type: $modelType")

        val tempFile = File(targetFile.parentFile, "${targetFile.name}.download")

        // Check if partial download exists for resume
        var downloadedBytes = if (tempFile.exists()) tempFile.length() else 0L

        val requestBuilder = Request.Builder().url(url)
        if (downloadedBytes > 0) {
            requestBuilder.addHeader("Range", "bytes=$downloadedBytes-")
        }

        val response = client.newCall(requestBuilder.build()).execute()

        if (!response.isSuccessful && response.code != 206) {
            response.close()
            throw RuntimeException("Download failed: HTTP ${response.code}")
        }

        val body = response.body ?: throw RuntimeException("Empty response body")
        val contentLength = body.contentLength()
        val totalBytes = if (response.code == 206) {
            // Partial content - total is downloaded + remaining
            downloadedBytes + contentLength
        } else {
            // Full download
            downloadedBytes = 0 // Restart
            contentLength
        }

        val inputStream = body.byteStream()
        val outputStream = FileOutputStream(tempFile, downloadedBytes > 0)
        val buffer = ByteArray(8192)

        try {
            var bytesRead: Int
            while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                outputStream.write(buffer, 0, bytesRead)
                downloadedBytes += bytesRead
                if (totalBytes > 0) {
                    onProgress(downloadedBytes.toFloat() / totalBytes.toFloat())
                }
            }
            outputStream.flush()
        } finally {
            inputStream.close()
            outputStream.close()
            response.close()
        }

        // Rename temp file to final name
        if (!tempFile.renameTo(targetFile)) {
            // Fallback: copy and delete
            tempFile.copyTo(targetFile, overwrite = true)
            tempFile.delete()
        }

        onProgress(1.0f)
    }
}

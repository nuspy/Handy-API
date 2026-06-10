package com.handy.voice.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Records audio from the microphone at 16kHz mono for STT processing.
 *
 * Usage:
 *   recorder.start()
 *   // ... user speaks ...
 *   val audioData = recorder.stopAndGetData()
 */
@Singleton
class AudioRecorder @Inject constructor() {

    companion object {
        const val SAMPLE_RATE = 16000
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_FLOAT
    }

    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private val audioBuffer = mutableListOf<Float>()
    private var recordingThread: Thread? = null

    @SuppressLint("MissingPermission")
    fun start() {
        if (isRecording) return

        val bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
            .coerceAtLeast(SAMPLE_RATE) // At least 1 second buffer

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT,
            bufferSize * 2,
        )

        audioBuffer.clear()
        isRecording = true
        audioRecord?.startRecording()

        recordingThread = Thread {
            val buffer = FloatArray(bufferSize / 4)
            while (isRecording) {
                val read = audioRecord?.read(buffer, 0, buffer.size, AudioRecord.READ_BLOCKING) ?: -1
                if (read > 0) {
                    synchronized(audioBuffer) {
                        for (i in 0 until read) {
                            audioBuffer.add(buffer[i])
                        }
                    }
                }
            }
        }.also { it.start() }
    }

    /**
     * Stop recording and return the captured audio as float32 PCM samples.
     * Returns null if no data was recorded.
     */
    fun stopAndGetData(): FloatArray? {
        isRecording = false
        recordingThread?.join(1000)
        recordingThread = null

        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null

        val data: FloatArray?
        synchronized(audioBuffer) {
            data = if (audioBuffer.isEmpty()) null else audioBuffer.toFloatArray()
            audioBuffer.clear()
        }
        return data
    }

    fun isRecording(): Boolean = isRecording
}

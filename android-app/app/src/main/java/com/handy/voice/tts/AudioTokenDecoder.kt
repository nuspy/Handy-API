package com.handy.voice.tts

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import com.handy.voice.models.ModelManager
import dagger.hilt.android.qualifiers.ApplicationContext
import java.nio.FloatBuffer
import java.nio.LongBuffer
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Decodes audio tokens (produced by Qwen3-TTS LLM) into PCM waveform
 * using the Qwen3-TTS-Tokenizer-12Hz ONNX model.
 *
 * The tokenizer uses a multi-codebook design with 16 layers at 12Hz frame rate.
 * Input: audio token IDs (from LLM generation)
 * Output: PCM float32 waveform at 24kHz
 *
 * NOTE: The exact input/output shapes depend on the exported ONNX model.
 * Run verify_qwen3_tts.py first to determine the correct shapes.
 */
@Singleton
class AudioTokenDecoder @Inject constructor(
    @ApplicationContext private val context: Context,
    private val modelManager: ModelManager,
) {
    private var session: OrtSession? = null
    private val env = OrtEnvironment.getEnvironment()

    companion object {
        private const val SAMPLE_RATE = 24000
        private const val FRAME_RATE = 12 // 12Hz token rate
        private const val NUM_CODEBOOKS = 16
        private const val SAMPLES_PER_FRAME = SAMPLE_RATE / FRAME_RATE // 2000 samples per frame
    }

    fun initialize(modelPath: String? = null): Boolean {
        release()
        val path = modelPath ?: modelManager.getModelPath("tts_decoder") ?: return false

        return try {
            val options = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setIntraOpNumThreads(4)
                // Enable NNAPI for Snapdragon NPU acceleration
                // addNnapi() // Uncomment when NNAPI provider is available
            }
            session = env.createSession(path, options)
            true
        } catch (e: Exception) {
            android.util.Log.e("AudioTokenDecoder", "Failed to load ONNX model", e)
            false
        }
    }

    /**
     * Decode audio token IDs into PCM waveform.
     *
     * @param tokenIds Array of audio token IDs from LLM generation.
     *                 These represent multi-codebook indices.
     * @return Float array of PCM samples at 24kHz, or null on failure.
     */
    fun decode(tokenIds: IntArray): FloatArray? {
        val sess = session
        if (sess == null) {
            if (!initialize()) return null
        }
        val activeSession = session ?: return null

        return try {
            // Reshape token IDs into codebook format
            // The LLM generates interleaved tokens: [cb0_t0, cb1_t0, ..., cb15_t0, cb0_t1, ...]
            val numFrames = tokenIds.size / NUM_CODEBOOKS
            if (numFrames == 0) return null

            val inputShape = longArrayOf(1, NUM_CODEBOOKS.toLong(), numFrames.toLong())
            val inputData = LongArray(tokenIds.size) { tokenIds[it].toLong() }
            val inputBuffer = LongBuffer.wrap(inputData)

            val inputTensor = OnnxTensor.createTensor(env, inputBuffer, inputShape)
            val inputs = mapOf("audio_tokens" to inputTensor)

            val results = activeSession.run(inputs)
            val outputTensor = results[0] as OnnxTensor

            // Extract waveform
            val waveform = outputTensor.floatBuffer
            val pcmData = FloatArray(waveform.remaining())
            waveform.get(pcmData)

            inputTensor.close()
            results.close()

            pcmData
        } catch (e: Exception) {
            android.util.Log.e("AudioTokenDecoder", "Decode failed", e)
            null
        }
    }

    fun release() {
        session?.close()
        session = null
    }

    fun isLoaded(): Boolean = session != null
}

package com.handy.voice.audio

import javax.inject.Inject
import javax.inject.Singleton

/**
 * Audio processing utilities for resampling, normalization, and format conversion.
 */
@Singleton
class AudioProcessor @Inject constructor() {

    /**
     * Resample audio from source to target sample rate using linear interpolation.
     */
    fun resample(
        input: FloatArray,
        sourceSampleRate: Int,
        targetSampleRate: Int,
    ): FloatArray {
        if (sourceSampleRate == targetSampleRate) return input

        val ratio = targetSampleRate.toDouble() / sourceSampleRate.toDouble()
        val outputLength = (input.size * ratio).toInt()
        val output = FloatArray(outputLength)

        for (i in output.indices) {
            val srcIdx = i / ratio
            val srcIdxFloor = srcIdx.toInt()
            val frac = (srcIdx - srcIdxFloor).toFloat()

            val s0 = input[srcIdxFloor.coerceIn(0, input.size - 1)]
            val s1 = input[(srcIdxFloor + 1).coerceIn(0, input.size - 1)]
            output[i] = s0 + frac * (s1 - s0)
        }
        return output
    }

    /**
     * Normalize audio to have peak amplitude of targetPeak.
     */
    fun normalize(audio: FloatArray, targetPeak: Float = 0.95f): FloatArray {
        val maxAbs = audio.maxOfOrNull { kotlin.math.abs(it) } ?: return audio
        if (maxAbs < 1e-6f) return audio

        val scale = targetPeak / maxAbs
        return FloatArray(audio.size) { audio[it] * scale }
    }

    /**
     * Trim silence from the beginning and end of audio.
     *
     * @param threshold Amplitude below which is considered silence
     * @param minSamples Minimum number of consecutive samples above threshold
     */
    fun trimSilence(
        audio: FloatArray,
        threshold: Float = 0.01f,
        minSamples: Int = 160, // 10ms at 16kHz
    ): FloatArray {
        var start = 0
        var end = audio.size - 1

        // Find start
        var consecutiveAbove = 0
        for (i in audio.indices) {
            if (kotlin.math.abs(audio[i]) > threshold) {
                consecutiveAbove++
                if (consecutiveAbove >= minSamples) {
                    start = (i - minSamples).coerceAtLeast(0)
                    break
                }
            } else {
                consecutiveAbove = 0
            }
        }

        // Find end
        consecutiveAbove = 0
        for (i in audio.indices.reversed()) {
            if (kotlin.math.abs(audio[i]) > threshold) {
                consecutiveAbove++
                if (consecutiveAbove >= minSamples) {
                    end = (i + minSamples).coerceAtMost(audio.size - 1)
                    break
                }
            } else {
                consecutiveAbove = 0
            }
        }

        return if (start < end) audio.sliceArray(start..end) else audio
    }
}

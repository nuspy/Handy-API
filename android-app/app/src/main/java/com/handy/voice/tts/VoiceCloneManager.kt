package com.handy.voice.tts

import android.content.Context
import com.handy.voice.ui.VoiceProfile
import dagger.hilt.android.qualifiers.ApplicationContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Manages voice clone profiles.
 *
 * Each profile stores:
 * - A reference audio file (WAV, 16kHz mono, 3-10 seconds)
 * - Metadata (name, duration, creation date)
 *
 * Profiles are stored in app internal storage under "voice_profiles/".
 */
@Singleton
class VoiceCloneManager @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    private val profilesDir: File
        get() = File(context.filesDir, "voice_profiles").also { it.mkdirs() }

    private val metadataFile: File
        get() = File(profilesDir, "profiles.json")

    fun getProfiles(): List<VoiceProfile> {
        if (!metadataFile.exists()) return emptyList()

        return try {
            val json = JSONArray(metadataFile.readText())
            (0 until json.length()).map { i ->
                val obj = json.getJSONObject(i)
                VoiceProfile(
                    id = obj.getString("id"),
                    name = obj.getString("name"),
                    durationSeconds = obj.getDouble("duration").toFloat(),
                    createdAt = obj.optLong("createdAt", 0),
                )
            }
        } catch (e: Exception) {
            emptyList()
        }
    }

    /**
     * Create a new voice profile from recorded audio.
     *
     * @param name User-friendly name for the voice
     * @param audioData Raw PCM float32 samples at 16kHz
     * @param durationSeconds Duration of the recording
     */
    fun createProfile(
        name: String,
        audioData: FloatArray,
        durationSeconds: Float,
    ): VoiceProfile? {
        return try {
            val id = UUID.randomUUID().toString()
            val audioFile = File(profilesDir, "$id.wav")

            // Save as WAV file
            saveAsWav(audioFile, audioData, sampleRate = 16000)

            val profile = VoiceProfile(
                id = id,
                name = name,
                durationSeconds = durationSeconds,
            )

            // Update metadata
            val profiles = getProfiles().toMutableList()
            profiles.add(profile)
            saveMetadata(profiles)

            profile
        } catch (e: Exception) {
            android.util.Log.e("VoiceCloneManager", "Failed to create profile", e)
            null
        }
    }

    fun deleteProfile(profileId: String) {
        File(profilesDir, "$profileId.wav").delete()
        val profiles = getProfiles().filter { it.id != profileId }
        saveMetadata(profiles)
    }

    fun getProfileAudioPath(profileId: String): String? {
        val file = File(profilesDir, "$profileId.wav")
        return if (file.exists()) file.absolutePath else null
    }

    private fun saveMetadata(profiles: List<VoiceProfile>) {
        val json = JSONArray()
        profiles.forEach { profile ->
            json.put(JSONObject().apply {
                put("id", profile.id)
                put("name", profile.name)
                put("duration", profile.durationSeconds)
                put("createdAt", profile.createdAt)
            })
        }
        metadataFile.writeText(json.toString(2))
    }

    private fun saveAsWav(file: File, samples: FloatArray, sampleRate: Int) {
        val numChannels = 1
        val bitsPerSample = 16
        val byteRate = sampleRate * numChannels * bitsPerSample / 8
        val blockAlign = numChannels * bitsPerSample / 8
        val dataSize = samples.size * 2 // 16-bit = 2 bytes per sample

        FileOutputStream(file).use { fos ->
            val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).apply {
                // RIFF header
                put("RIFF".toByteArray())
                putInt(36 + dataSize)
                put("WAVE".toByteArray())

                // fmt chunk
                put("fmt ".toByteArray())
                putInt(16) // chunk size
                putShort(1) // PCM format
                putShort(numChannels.toShort())
                putInt(sampleRate)
                putInt(byteRate)
                putShort(blockAlign.toShort())
                putShort(bitsPerSample.toShort())

                // data chunk
                put("data".toByteArray())
                putInt(dataSize)
            }
            fos.write(header.array())

            // Write PCM data (convert float32 -> int16)
            val pcmBuffer = ByteBuffer.allocate(dataSize).order(ByteOrder.LITTLE_ENDIAN)
            for (sample in samples) {
                val clamped = sample.coerceIn(-1f, 1f)
                pcmBuffer.putShort((clamped * Short.MAX_VALUE).toInt().toShort())
            }
            fos.write(pcmBuffer.array())
        }
    }
}

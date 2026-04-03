package com.handy.voice.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.handy.voice.audio.AudioPlayer
import com.handy.voice.audio.AudioRecorder
import com.handy.voice.tts.QwenTTSEngine
import com.handy.voice.tts.VoiceCloneManager
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class VoiceCloneViewModel @Inject constructor(
    private val audioRecorder: AudioRecorder,
    private val ttsEngine: QwenTTSEngine,
    private val voiceCloneManager: VoiceCloneManager,
    private val audioPlayer: AudioPlayer,
) : ViewModel() {

    private val _voiceProfiles = MutableStateFlow<List<VoiceProfile>>(emptyList())
    val voiceProfiles = _voiceProfiles.asStateFlow()

    private val _isRecording = MutableStateFlow(false)
    val isRecording = _isRecording.asStateFlow()

    private val _recordingDuration = MutableStateFlow(0f)
    val recordingDuration = _recordingDuration.asStateFlow()

    private var durationJob: Job? = null

    init {
        loadProfiles()
    }

    private fun loadProfiles() {
        viewModelScope.launch(Dispatchers.IO) {
            _voiceProfiles.value = voiceCloneManager.getProfiles()
        }
    }

    fun startRecording() {
        _isRecording.value = true
        _recordingDuration.value = 0f
        audioRecorder.start()

        durationJob = viewModelScope.launch {
            while (_isRecording.value) {
                delay(100)
                _recordingDuration.value += 0.1f
            }
        }
    }

    fun stopRecording() {
        _isRecording.value = false
        durationJob?.cancel()
        // Audio data is held in audioRecorder until saveProfile or discardRecording
    }

    fun saveProfile(name: String) {
        viewModelScope.launch(Dispatchers.IO) {
            val audioData = audioRecorder.stopAndGetData() ?: return@launch
            val profile = voiceCloneManager.createProfile(
                name = name,
                audioData = audioData,
                durationSeconds = _recordingDuration.value,
            )
            if (profile != null) {
                _voiceProfiles.value = _voiceProfiles.value + profile
            }
        }
    }

    fun discardRecording() {
        audioRecorder.stopAndGetData() // discard
    }

    fun deleteProfile(profileId: String) {
        viewModelScope.launch(Dispatchers.IO) {
            voiceCloneManager.deleteProfile(profileId)
            _voiceProfiles.value = _voiceProfiles.value.filter { it.id != profileId }
        }
    }

    fun testVoiceProfile(profileId: String, text: String) {
        viewModelScope.launch(Dispatchers.IO) {
            val audioData = ttsEngine.synthesize(
                text = text,
                language = "auto",
                voiceProfile = profileId,
            )
            if (audioData != null) {
                audioPlayer.play(audioData, sampleRate = 24000)
            }
        }
    }
}

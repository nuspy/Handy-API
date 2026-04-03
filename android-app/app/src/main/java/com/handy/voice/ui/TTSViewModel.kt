package com.handy.voice.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.handy.voice.audio.AudioPlayer
import com.handy.voice.tts.QwenTTSEngine
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class TTSViewModel @Inject constructor(
    private val ttsEngine: QwenTTSEngine,
    private val audioPlayer: AudioPlayer,
) : ViewModel() {

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating = _isGenerating.asStateFlow()

    private val _progress = MutableStateFlow(0f)
    val progress = _progress.asStateFlow()

    fun generateSpeech(text: String, language: String, voiceProfile: String? = null) {
        viewModelScope.launch(Dispatchers.IO) {
            _isGenerating.value = true
            _progress.value = 0f

            try {
                val audioData = ttsEngine.synthesize(
                    text = text,
                    language = language,
                    voiceProfile = voiceProfile,
                    onProgress = { _progress.value = it },
                )

                if (audioData != null) {
                    audioPlayer.play(audioData, sampleRate = 24000)
                }
            } finally {
                _isGenerating.value = false
                _progress.value = 0f
            }
        }
    }

    fun stopPlayback() {
        audioPlayer.stop()
        _isGenerating.value = false
    }
}

package com.handy.voice.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.handy.voice.audio.AudioRecorder
import com.handy.voice.stt.WhisperEngine
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class STTViewModel @Inject constructor(
    private val audioRecorder: AudioRecorder,
    private val whisperEngine: WhisperEngine,
) : ViewModel() {

    fun startRecording() {
        audioRecorder.start()
    }

    fun stopRecording(onResult: (String) -> Unit) {
        viewModelScope.launch(Dispatchers.IO) {
            val audioData = audioRecorder.stopAndGetData()
            if (audioData != null) {
                val result = whisperEngine.transcribe(audioData)
                launch(Dispatchers.Main) {
                    onResult(result)
                }
            } else {
                launch(Dispatchers.Main) {
                    onResult("[error: no audio recorded]")
                }
            }
        }
    }
}

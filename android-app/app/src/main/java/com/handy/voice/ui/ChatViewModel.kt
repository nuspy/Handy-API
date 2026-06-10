package com.handy.voice.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.handy.voice.audio.AudioPlayer
import com.handy.voice.audio.AudioRecorder
import com.handy.voice.stt.WhisperEngine
import com.handy.voice.tts.LLMEngine
import com.handy.voice.tts.QwenTTSEngine
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class ChatViewModel @Inject constructor(
    private val llmEngine: LLMEngine,
    private val ttsEngine: QwenTTSEngine,
    private val whisperEngine: WhisperEngine,
    private val audioRecorder: AudioRecorder,
    private val audioPlayer: AudioPlayer,
) : ViewModel() {

    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages = _messages.asStateFlow()

    private val _isRecording = MutableStateFlow(false)
    val isRecording = _isRecording.asStateFlow()

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating = _isGenerating.asStateFlow()

    fun sendMessage(text: String) {
        viewModelScope.launch(Dispatchers.IO) {
            // Add user message
            _messages.value = _messages.value + ChatMessage(content = text, isUser = true)

            // Add loading placeholder
            _messages.value = _messages.value + ChatMessage(
                content = "",
                isUser = false,
                isLoading = true,
            )
            _isGenerating.value = true

            try {
                // Build conversation history for context
                val history = _messages.value
                    .filter { !it.isLoading }
                    .joinToString("\n") { msg ->
                        if (msg.isUser) "User: ${msg.content}" else "Assistant: ${msg.content}"
                    }

                val prompt = buildString {
                    append("<|system|>You are a helpful AI assistant. Respond concisely.<|end|>\n")
                    append(history)
                    append("\nAssistant:")
                }

                val response = llmEngine.generate(prompt, maxTokens = 512)

                // Replace loading message with actual response
                _messages.value = _messages.value.dropLast(1) + ChatMessage(
                    content = response.trim(),
                    isUser = false,
                )
            } catch (e: Exception) {
                _messages.value = _messages.value.dropLast(1) + ChatMessage(
                    content = "[Error: ${e.message}]",
                    isUser = false,
                )
            } finally {
                _isGenerating.value = false
            }
        }
    }

    fun startRecording() {
        _isRecording.value = true
        audioRecorder.start()
    }

    fun stopRecording(onResult: (String) -> Unit) {
        _isRecording.value = false
        viewModelScope.launch(Dispatchers.IO) {
            val audioData = audioRecorder.stopAndGetData()
            if (audioData != null) {
                val text = whisperEngine.transcribe(audioData)
                launch(Dispatchers.Main) {
                    onResult(text)
                }
            }
        }
    }

    fun speakMessage(text: String) {
        viewModelScope.launch(Dispatchers.IO) {
            val audioData = ttsEngine.synthesize(text = text, language = "auto")
            if (audioData != null) {
                audioPlayer.play(audioData, sampleRate = 24000)
            }
        }
    }
}

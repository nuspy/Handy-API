package com.handy.voice.ui

import android.app.ActivityManager
import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.handy.voice.models.ModelDownloader
import com.handy.voice.models.ModelManager
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class SettingsViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
    private val modelManager: ModelManager,
    private val modelDownloader: ModelDownloader,
) : ViewModel() {

    private val _sttModelStatus = MutableStateFlow(ModelStatus.NOT_DOWNLOADED)
    val sttModelStatus = _sttModelStatus.asStateFlow()

    private val _ttsModelStatus = MutableStateFlow(ModelStatus.NOT_DOWNLOADED)
    val ttsModelStatus = _ttsModelStatus.asStateFlow()

    private val _chatModelStatus = MutableStateFlow(ModelStatus.NOT_DOWNLOADED)
    val chatModelStatus = _chatModelStatus.asStateFlow()

    private val _downloadProgress = MutableStateFlow<Map<String, Float>>(emptyMap())
    val downloadProgress = _downloadProgress.asStateFlow()

    init {
        checkModelStatuses()
    }

    private fun checkModelStatuses() {
        viewModelScope.launch(Dispatchers.IO) {
            _sttModelStatus.value = if (modelManager.isModelAvailable("stt")) ModelStatus.READY else ModelStatus.NOT_DOWNLOADED
            _ttsModelStatus.value = if (modelManager.isModelAvailable("tts")) ModelStatus.READY else ModelStatus.NOT_DOWNLOADED
            _chatModelStatus.value = if (modelManager.isModelAvailable("chat")) ModelStatus.READY else ModelStatus.NOT_DOWNLOADED
        }
    }

    fun downloadModel(modelType: String) {
        val statusFlow = when (modelType) {
            "stt" -> _sttModelStatus
            "tts" -> _ttsModelStatus
            "chat" -> _chatModelStatus
            else -> return
        }

        statusFlow.value = ModelStatus.DOWNLOADING

        viewModelScope.launch(Dispatchers.IO) {
            try {
                modelDownloader.download(
                    modelType = modelType,
                    onProgress = { progress ->
                        _downloadProgress.value = _downloadProgress.value + (modelType to progress)
                    },
                )
                statusFlow.value = ModelStatus.READY
                _downloadProgress.value = _downloadProgress.value - modelType
            } catch (e: Exception) {
                statusFlow.value = ModelStatus.ERROR
                _downloadProgress.value = _downloadProgress.value - modelType
            }
        }
    }

    fun getAvailableRam(): String {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        val availMb = memInfo.availMem / (1024 * 1024)
        val totalMb = memInfo.totalMem / (1024 * 1024)
        return "${availMb}MB / ${totalMb}MB"
    }
}

package com.handy.voice.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel

@Composable
fun SettingsScreen(viewModel: SettingsViewModel = hiltViewModel()) {
    val sttModelStatus by viewModel.sttModelStatus.collectAsState()
    val ttsModelStatus by viewModel.ttsModelStatus.collectAsState()
    val chatModelStatus by viewModel.chatModelStatus.collectAsState()
    val downloadProgress by viewModel.downloadProgress.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
    ) {
        Text(
            text = "Settings",
            style = MaterialTheme.typography.headlineMedium,
            modifier = Modifier.padding(bottom = 16.dp),
        )

        // Models Section
        Text(
            text = "Models",
            style = MaterialTheme.typography.titleLarge,
            modifier = Modifier.padding(bottom = 8.dp),
        )

        ModelCard(
            name = "Whisper Small (STT)",
            description = "Speech-to-text model. ~500MB",
            status = sttModelStatus,
            downloadProgress = downloadProgress["stt"],
            onDownload = { viewModel.downloadModel("stt") },
        )

        Spacer(modifier = Modifier.height(8.dp))

        ModelCard(
            name = "Qwen3-TTS 0.6B (TTS)",
            description = "Text-to-speech + voice cloning. ~400MB (Q4_K_M)",
            status = ttsModelStatus,
            downloadProgress = downloadProgress["tts"],
            onDownload = { viewModel.downloadModel("tts") },
        )

        Spacer(modifier = Modifier.height(8.dp))

        ModelCard(
            name = "Chat LLM (Optional)",
            description = "LLM for AI chat. Configurable model.",
            status = chatModelStatus,
            downloadProgress = downloadProgress["chat"],
            onDownload = { viewModel.downloadModel("chat") },
        )

        Spacer(modifier = Modifier.height(16.dp))
        HorizontalDivider()
        Spacer(modifier = Modifier.height(16.dp))

        // System Info
        Text(
            text = "System Info",
            style = MaterialTheme.typography.titleLarge,
            modifier = Modifier.padding(bottom = 8.dp),
        )

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            ),
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                InfoRow("App version", "0.1.0")
                InfoRow("Device", android.os.Build.MODEL)
                InfoRow("Android", "API ${android.os.Build.VERSION.SDK_INT}")
                InfoRow("Available RAM", viewModel.getAvailableRam())
            }
        }
    }
}

@Composable
private fun ModelCard(
    name: String,
    description: String,
    status: ModelStatus,
    downloadProgress: Float?,
    onDownload: () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(text = name, style = MaterialTheme.typography.titleMedium)
                    Text(
                        text = description,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }

                when (status) {
                    ModelStatus.NOT_DOWNLOADED -> {
                        OutlinedButton(onClick = onDownload) {
                            Icon(Icons.Default.Download, contentDescription = null, modifier = Modifier.size(18.dp))
                            Spacer(modifier = Modifier.width(4.dp))
                            Text("Download")
                        }
                    }
                    ModelStatus.DOWNLOADING -> {
                        CircularProgressIndicator(modifier = Modifier.size(24.dp))
                    }
                    ModelStatus.READY -> {
                        Icon(
                            Icons.Default.Check,
                            contentDescription = "Ready",
                            tint = MaterialTheme.colorScheme.primary,
                        )
                    }
                    ModelStatus.ERROR -> {
                        OutlinedButton(onClick = onDownload) {
                            Text("Retry")
                        }
                    }
                }
            }

            if (downloadProgress != null && status == ModelStatus.DOWNLOADING) {
                Spacer(modifier = Modifier.height(8.dp))
                LinearProgressIndicator(
                    progress = { downloadProgress },
                    modifier = Modifier.fillMaxWidth(),
                )
                Text(
                    text = "${(downloadProgress * 100).toInt()}%",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
    }
}

@Composable
private fun InfoRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.weight(1f),
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
        )
    }
}

enum class ModelStatus {
    NOT_DOWNLOADED,
    DOWNLOADING,
    READY,
    ERROR,
}

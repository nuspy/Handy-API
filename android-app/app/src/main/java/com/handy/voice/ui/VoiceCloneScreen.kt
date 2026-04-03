package com.handy.voice.ui

import android.Manifest
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel

data class VoiceProfile(
    val id: String,
    val name: String,
    val durationSeconds: Float,
    val createdAt: Long = System.currentTimeMillis(),
)

@Composable
fun VoiceCloneScreen(viewModel: VoiceCloneViewModel = hiltViewModel()) {
    val profiles by viewModel.voiceProfiles.collectAsState()
    val isRecording by viewModel.isRecording.collectAsState()
    val recordingDuration by viewModel.recordingDuration.collectAsState()
    var showAddDialog by remember { mutableStateOf(false) }
    var newProfileName by remember { mutableStateOf("") }
    var testText by remember { mutableStateOf("") }
    var hasPermission by remember { mutableStateOf(false) }
    val context = LocalContext.current

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasPermission = granted
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
    ) {
        Text(
            text = "Voice Cloning",
            style = MaterialTheme.typography.headlineMedium,
            modifier = Modifier.padding(bottom = 8.dp),
        )

        Text(
            text = "Record at least 3 seconds of voice to create a clone profile",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(bottom = 16.dp),
        )

        // Voice profiles list
        LazyColumn(
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            items(profiles) { profile ->
                VoiceProfileCard(
                    profile = profile,
                    testText = testText,
                    onTestTextChange = { testText = it },
                    onPlay = {
                        if (testText.isNotBlank()) {
                            viewModel.testVoiceProfile(profile.id, testText)
                        } else {
                            Toast.makeText(context, "Enter test text first", Toast.LENGTH_SHORT).show()
                        }
                    },
                    onDelete = { viewModel.deleteProfile(profile.id) },
                )
            }

            if (profiles.isEmpty()) {
                item {
                    Text(
                        text = "No voice profiles yet. Tap + to create one.",
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(32.dp),
                    )
                }
            }
        }

        // Recording indicator
        if (isRecording) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.errorContainer
                ),
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Text(
                        text = "Recording: ${String.format("%.1f", recordingDuration)}s",
                        style = MaterialTheme.typography.titleMedium,
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    LinearProgressIndicator(
                        progress = { (recordingDuration / 10f).coerceIn(0f, 1f) },
                        modifier = Modifier.fillMaxWidth(),
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = if (recordingDuration < 3f) "Keep recording (min 3s)" else "Good! You can stop now",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Button(onClick = {
                        if (recordingDuration >= 3f) {
                            showAddDialog = true
                            viewModel.stopRecording()
                        }
                    }) {
                        Icon(Icons.Default.Stop, contentDescription = null)
                        Text("Stop", modifier = Modifier.padding(start = 4.dp))
                    }
                }
            }
            Spacer(modifier = Modifier.height(16.dp))
        }

        // FAB to add new profile
        if (!isRecording) {
            FloatingActionButton(
                onClick = {
                    if (!hasPermission) {
                        permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                        return@FloatingActionButton
                    }
                    viewModel.startRecording()
                },
                modifier = Modifier.align(Alignment.End),
            ) {
                Icon(Icons.Default.Add, contentDescription = "New voice profile")
            }
        }
    }

    // Name dialog
    if (showAddDialog) {
        AlertDialog(
            onDismissRequest = { showAddDialog = false },
            title = { Text("Name this voice") },
            text = {
                OutlinedTextField(
                    value = newProfileName,
                    onValueChange = { newProfileName = it },
                    label = { Text("Voice name") },
                    placeholder = { Text("e.g., My Voice") },
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        if (newProfileName.isNotBlank()) {
                            viewModel.saveProfile(newProfileName)
                            newProfileName = ""
                            showAddDialog = false
                        }
                    },
                ) {
                    Text("Save")
                }
            },
            dismissButton = {
                TextButton(onClick = {
                    showAddDialog = false
                    viewModel.discardRecording()
                }) {
                    Text("Cancel")
                }
            },
        )
    }
}

@Composable
private fun VoiceProfileCard(
    profile: VoiceProfile,
    testText: String,
    onTestTextChange: (String) -> Unit,
    onPlay: () -> Unit,
    onDelete: () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column {
                    Text(
                        text = profile.name,
                        style = MaterialTheme.typography.titleMedium,
                    )
                    Text(
                        text = "${String.format("%.1f", profile.durationSeconds)}s sample",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Row {
                    IconButton(onClick = onPlay) {
                        Icon(Icons.Default.PlayArrow, contentDescription = "Test voice")
                    }
                    IconButton(onClick = onDelete) {
                        Icon(
                            Icons.Default.Delete,
                            contentDescription = "Delete profile",
                            tint = MaterialTheme.colorScheme.error,
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            OutlinedTextField(
                value = testText,
                onValueChange = onTestTextChange,
                label = { Text("Test text") },
                placeholder = { Text("Enter text to test this voice...") },
                modifier = Modifier.fillMaxWidth(),
                maxLines = 2,
            )
        }
    }
}

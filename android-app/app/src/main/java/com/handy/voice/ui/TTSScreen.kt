package com.handy.voice.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TTSScreen(viewModel: TTSViewModel = hiltViewModel()) {
    var inputText by remember { mutableStateOf("") }
    var selectedLanguage by remember { mutableStateOf("en") }
    var languageExpanded by remember { mutableStateOf(false) }
    val isGenerating by viewModel.isGenerating.collectAsState()
    val progress by viewModel.progress.collectAsState()

    val languages = listOf(
        "en" to "English",
        "it" to "Italiano",
        "zh" to "Chinese",
        "ja" to "Japanese",
        "ko" to "Korean",
        "de" to "Deutsch",
        "fr" to "Francais",
        "es" to "Espanol",
        "pt" to "Portugues",
        "ru" to "Russian",
    )

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text = "Text to Speech",
            style = MaterialTheme.typography.headlineMedium,
            modifier = Modifier.padding(bottom = 16.dp),
        )

        // Text input
        OutlinedTextField(
            value = inputText,
            onValueChange = { inputText = it },
            label = { Text("Enter text to speak") },
            placeholder = { Text("Type or paste text here...") },
            modifier = Modifier
                .fillMaxWidth()
                .height(200.dp),
            maxLines = 10,
        )

        Spacer(modifier = Modifier.height(16.dp))

        // Language selector
        ExposedDropdownMenuBox(
            expanded = languageExpanded,
            onExpandedChange = { languageExpanded = it },
            modifier = Modifier.fillMaxWidth(),
        ) {
            OutlinedTextField(
                value = languages.find { it.first == selectedLanguage }?.second ?: "English",
                onValueChange = {},
                readOnly = true,
                label = { Text("Language") },
                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = languageExpanded) },
                modifier = Modifier
                    .menuAnchor(MenuAnchorType.PrimaryNotEditable)
                    .fillMaxWidth(),
            )
            ExposedDropdownMenu(
                expanded = languageExpanded,
                onDismissRequest = { languageExpanded = false },
            ) {
                languages.forEach { (code, name) ->
                    DropdownMenuItem(
                        text = { Text(name) },
                        onClick = {
                            selectedLanguage = code
                            languageExpanded = false
                        },
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Progress indicator
        if (isGenerating) {
            LinearProgressIndicator(
                progress = { progress },
                modifier = Modifier.fillMaxWidth(),
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Generating speech...",
                style = MaterialTheme.typography.bodySmall,
            )
            Spacer(modifier = Modifier.height(8.dp))
        }

        // Generate button
        Button(
            onClick = {
                if (isGenerating) {
                    viewModel.stopPlayback()
                } else if (inputText.isNotBlank()) {
                    viewModel.generateSpeech(inputText, selectedLanguage)
                }
            },
            enabled = inputText.isNotBlank() || isGenerating,
            modifier = Modifier.fillMaxWidth(),
            colors = if (isGenerating)
                ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error)
            else
                ButtonDefaults.buttonColors(),
        ) {
            Icon(
                imageVector = if (isGenerating) Icons.Default.Stop else Icons.Default.PlayArrow,
                contentDescription = null,
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(if (isGenerating) "Stop" else "Speak")
        }
    }
}

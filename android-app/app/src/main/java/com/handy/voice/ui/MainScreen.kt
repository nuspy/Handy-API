package com.handy.voice.ui

import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.RecordVoiceOver
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.VolumeUp
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController

enum class Screen(val route: String, val label: String, val icon: ImageVector) {
    STT("stt", "STT", Icons.Default.Mic),
    TTS("tts", "TTS", Icons.Default.VolumeUp),
    Chat("chat", "Chat", Icons.Default.Chat),
    VoiceClone("voice_clone", "Clone", Icons.Default.RecordVoiceOver),
    Settings("settings", "Settings", Icons.Default.Settings),
}

@Composable
fun HandyVoiceApp() {
    val navController = rememberNavController()

    Scaffold(
        bottomBar = {
            NavigationBar {
                val navBackStackEntry by navController.currentBackStackEntryAsState()
                val currentDestination = navBackStackEntry?.destination

                Screen.entries.forEach { screen ->
                    NavigationBarItem(
                        icon = { Icon(screen.icon, contentDescription = screen.label) },
                        label = { Text(screen.label) },
                        selected = currentDestination?.hierarchy?.any { it.route == screen.route } == true,
                        onClick = {
                            navController.navigate(screen.route) {
                                popUpTo(navController.graph.findStartDestination().id) {
                                    saveState = true
                                }
                                launchSingleTop = true
                                restoreState = true
                            }
                        }
                    )
                }
            }
        }
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = Screen.STT.route,
            modifier = Modifier.padding(innerPadding),
        ) {
            composable(Screen.STT.route) { STTScreen() }
            composable(Screen.TTS.route) { TTSScreen() }
            composable(Screen.Chat.route) { ChatScreen() }
            composable(Screen.VoiceClone.route) { VoiceCloneScreen() }
            composable(Screen.Settings.route) { SettingsScreen() }
        }
    }
}

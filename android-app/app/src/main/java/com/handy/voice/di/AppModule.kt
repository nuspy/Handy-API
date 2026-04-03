package com.handy.voice.di

import android.content.Context
import com.handy.voice.audio.AudioPlayer
import com.handy.voice.audio.AudioProcessor
import com.handy.voice.audio.AudioRecorder
import com.handy.voice.models.ModelDownloader
import com.handy.voice.models.ModelManager
import com.handy.voice.stt.WhisperEngine
import com.handy.voice.tts.AudioTokenDecoder
import com.handy.voice.tts.LLMEngine
import com.handy.voice.tts.QwenTTSEngine
import com.handy.voice.tts.VoiceCloneManager
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    fun provideModelManager(@ApplicationContext context: Context): ModelManager =
        ModelManager(context)

    @Provides
    @Singleton
    fun provideModelDownloader(
        @ApplicationContext context: Context,
        modelManager: ModelManager,
    ): ModelDownloader = ModelDownloader(context, modelManager)

    @Provides
    @Singleton
    fun provideAudioRecorder(): AudioRecorder = AudioRecorder()

    @Provides
    @Singleton
    fun provideAudioPlayer(): AudioPlayer = AudioPlayer()

    @Provides
    @Singleton
    fun provideAudioProcessor(): AudioProcessor = AudioProcessor()

    @Provides
    @Singleton
    fun provideWhisperEngine(
        @ApplicationContext context: Context,
        modelManager: ModelManager,
    ): WhisperEngine = WhisperEngine(context, modelManager)

    @Provides
    @Singleton
    fun provideAudioTokenDecoder(
        @ApplicationContext context: Context,
        modelManager: ModelManager,
    ): AudioTokenDecoder = AudioTokenDecoder(context, modelManager)

    @Provides
    @Singleton
    fun provideQwenTTSEngine(
        @ApplicationContext context: Context,
        modelManager: ModelManager,
        audioTokenDecoder: AudioTokenDecoder,
    ): QwenTTSEngine = QwenTTSEngine(context, modelManager, audioTokenDecoder)

    @Provides
    @Singleton
    fun provideLLMEngine(
        @ApplicationContext context: Context,
        modelManager: ModelManager,
    ): LLMEngine = LLMEngine(context, modelManager)

    @Provides
    @Singleton
    fun provideVoiceCloneManager(
        @ApplicationContext context: Context,
    ): VoiceCloneManager = VoiceCloneManager(context)
}

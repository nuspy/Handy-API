#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>

#define TAG "WhisperJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#ifdef WHISPER_AVAILABLE
#include "whisper.h"
#endif

extern "C" {

// ============================================================
// Whisper Context Management
// ============================================================

JNIEXPORT jlong JNICALL
Java_com_handy_voice_stt_WhisperEngine_nativeInit(
    JNIEnv *env, jobject thiz, jstring model_path) {
#ifdef WHISPER_AVAILABLE
    const char *path = env->GetStringUTFChars(model_path, nullptr);
    LOGI("Loading whisper model: %s", path);

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true; // Enable GPU if available (Vulkan on Android)

    struct whisper_context *ctx = whisper_init_from_file_with_params(path, cparams);
    env->ReleaseStringUTFChars(model_path, path);

    if (ctx == nullptr) {
        LOGE("Failed to initialize whisper context");
        return 0;
    }

    LOGI("Whisper model loaded successfully");
    return reinterpret_cast<jlong>(ctx);
#else
    LOGE("Whisper not available - build without WHISPER_AVAILABLE");
    return 0;
#endif
}

JNIEXPORT void JNICALL
Java_com_handy_voice_stt_WhisperEngine_nativeRelease(
    JNIEnv *env, jobject thiz, jlong context_ptr) {
#ifdef WHISPER_AVAILABLE
    if (context_ptr != 0) {
        auto *ctx = reinterpret_cast<struct whisper_context *>(context_ptr);
        whisper_free(ctx);
        LOGI("Whisper context released");
    }
#endif
}

// ============================================================
// Transcription
// ============================================================

JNIEXPORT jstring JNICALL
Java_com_handy_voice_stt_WhisperEngine_nativeTranscribe(
    JNIEnv *env, jobject thiz, jlong context_ptr,
    jfloatArray audio_data, jstring language, jboolean translate) {
#ifdef WHISPER_AVAILABLE
    if (context_ptr == 0) {
        return env->NewStringUTF("[error: no context]");
    }

    auto *ctx = reinterpret_cast<struct whisper_context *>(context_ptr);

    // Get audio samples
    jfloat *samples = env->GetFloatArrayElements(audio_data, nullptr);
    jsize n_samples = env->GetArrayLength(audio_data);

    // Configure transcription parameters
    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_realtime = false;
    params.print_progress = false;
    params.print_timestamps = false;
    params.print_special = false;
    params.single_segment = false;
    params.no_timestamps = true;
    params.n_threads = 4;

    // Language setting
    if (language != nullptr) {
        const char *lang = env->GetStringUTFChars(language, nullptr);
        if (strcmp(lang, "auto") != 0) {
            params.language = lang;
        }
        // Note: we need to keep lang valid during whisper_full, so don't release yet
    }

    // Translation mode
    params.translate = translate;

    LOGI("Transcribing %d samples...", n_samples);

    int result = whisper_full(ctx, params, samples, n_samples);

    env->ReleaseFloatArrayElements(audio_data, samples, JNI_ABORT);

    if (result != 0) {
        LOGE("Transcription failed with code: %d", result);
        return env->NewStringUTF("[error: transcription failed]");
    }

    // Collect all segments
    std::string text;
    int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; i++) {
        const char *segment_text = whisper_full_get_segment_text(ctx, i);
        text += segment_text;
    }

    LOGI("Transcription complete: %d segments, %zu chars", n_segments, text.size());
    return env->NewStringUTF(text.c_str());
#else
    return env->NewStringUTF("[error: whisper not available]");
#endif
}

// ============================================================
// Model Info
// ============================================================

JNIEXPORT jstring JNICALL
Java_com_handy_voice_stt_WhisperEngine_nativeGetSystemInfo(
    JNIEnv *env, jobject thiz) {
#ifdef WHISPER_AVAILABLE
    const char *info = whisper_print_system_info();
    return env->NewStringUTF(info);
#else
    return env->NewStringUTF("whisper not available");
#endif
}

} // extern "C"

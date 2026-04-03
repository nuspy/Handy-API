#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>

#define TAG "LlamaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#ifdef LLAMA_AVAILABLE
#include "llama.h"
#include "ggml.h"
#endif

extern "C" {

// ============================================================
// Model + Context wrapper
// ============================================================

struct LlamaState {
#ifdef LLAMA_AVAILABLE
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    llama_sampler *sampler = nullptr;
#endif
    int n_ctx = 2048;
};

// ============================================================
// Initialization
// ============================================================

JNIEXPORT jlong JNICALL
Java_com_handy_voice_tts_QwenTTSEngine_nativeInit(
    JNIEnv *env, jobject thiz, jstring model_path, jint n_ctx, jint n_threads) {
#ifdef LLAMA_AVAILABLE
    const char *path = env->GetStringUTFChars(model_path, nullptr);
    LOGI("Loading llama model: %s", path);

    // Initialize backend
    llama_backend_init();

    // Load model
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // CPU-only for now; set >0 for Vulkan/GPU

    llama_model *model = llama_model_load_from_file(path, model_params);
    env->ReleaseStringUTFChars(model_path, path);

    if (model == nullptr) {
        LOGE("Failed to load llama model");
        return 0;
    }

    // Create context
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;

    llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        LOGE("Failed to create llama context");
        llama_model_free(model);
        return 0;
    }

    // Create sampler with default settings for TTS
    auto *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));

    auto *state = new LlamaState();
    state->model = model;
    state->ctx = ctx;
    state->sampler = sampler;
    state->n_ctx = n_ctx;

    LOGI("Llama model loaded successfully (ctx=%d, threads=%d)", n_ctx, n_threads);
    return reinterpret_cast<jlong>(state);
#else
    LOGE("llama.cpp not available");
    return 0;
#endif
}

JNIEXPORT void JNICALL
Java_com_handy_voice_tts_QwenTTSEngine_nativeRelease(
    JNIEnv *env, jobject thiz, jlong state_ptr) {
#ifdef LLAMA_AVAILABLE
    if (state_ptr != 0) {
        auto *state = reinterpret_cast<LlamaState *>(state_ptr);
        if (state->sampler) llama_sampler_free(state->sampler);
        if (state->ctx) llama_free(state->ctx);
        if (state->model) llama_model_free(state->model);
        delete state;
        llama_backend_free();
        LOGI("Llama state released");
    }
#endif
}

// ============================================================
// Token Generation (for TTS audio tokens)
// ============================================================

JNIEXPORT jintArray JNICALL
Java_com_handy_voice_tts_QwenTTSEngine_nativeGenerate(
    JNIEnv *env, jobject thiz, jlong state_ptr,
    jstring prompt, jint max_tokens, jfloat temperature) {
#ifdef LLAMA_AVAILABLE
    if (state_ptr == 0) {
        return env->NewIntArray(0);
    }

    auto *state = reinterpret_cast<LlamaState *>(state_ptr);
    const char *prompt_str = env->GetStringUTFChars(prompt, nullptr);

    LOGI("Generating tokens for prompt: %.80s...", prompt_str);

    // Tokenize prompt
    const llama_vocab *vocab = llama_model_get_vocab(state->model);
    int n_prompt_max = strlen(prompt_str) + 128;
    std::vector<llama_token> tokens(n_prompt_max);
    int n_tokens = llama_tokenize(vocab, prompt_str, strlen(prompt_str),
                                   tokens.data(), n_prompt_max, true, true);
    env->ReleaseStringUTFChars(prompt, prompt_str);

    if (n_tokens < 0) {
        LOGE("Tokenization failed");
        return env->NewIntArray(0);
    }
    tokens.resize(n_tokens);

    LOGI("Prompt tokenized: %d tokens", n_tokens);

    // Clear KV cache
    llama_kv_cache_clear(state->ctx);

    // Process prompt in batch
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        llama_batch_add(batch, tokens[i], i, {0}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(state->ctx, batch) != 0) {
        LOGE("Failed to decode prompt batch");
        llama_batch_free(batch);
        return env->NewIntArray(0);
    }

    // Generate tokens
    std::vector<jint> output_tokens;
    llama_token eos = llama_vocab_eos(vocab);
    int n_cur = n_tokens;

    for (int i = 0; i < max_tokens; i++) {
        llama_token new_token = llama_sampler_sample(state->sampler, state->ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) {
            LOGI("EOS reached at token %d", i);
            break;
        }

        output_tokens.push_back(static_cast<jint>(new_token));

        // Prepare next batch
        llama_batch_clear(batch);
        llama_batch_add(batch, new_token, n_cur, {0}, true);
        n_cur++;

        if (llama_decode(state->ctx, batch) != 0) {
            LOGE("Failed to decode at step %d", i);
            break;
        }
    }

    llama_batch_free(batch);

    LOGI("Generated %zu tokens", output_tokens.size());

    // Convert to Java int array
    jintArray result = env->NewIntArray(output_tokens.size());
    if (result != nullptr && !output_tokens.empty()) {
        env->SetIntArrayRegion(result, 0, output_tokens.size(), output_tokens.data());
    }
    return result;
#else
    return env->NewIntArray(0);
#endif
}

// ============================================================
// Chat Generation (for LLM chat)
// ============================================================

JNIEXPORT jstring JNICALL
Java_com_handy_voice_tts_QwenTTSEngine_nativeGenerateText(
    JNIEnv *env, jobject thiz, jlong state_ptr,
    jstring prompt, jint max_tokens, jfloat temperature) {
#ifdef LLAMA_AVAILABLE
    if (state_ptr == 0) {
        return env->NewStringUTF("");
    }

    auto *state = reinterpret_cast<LlamaState *>(state_ptr);
    const char *prompt_str = env->GetStringUTFChars(prompt, nullptr);

    // Tokenize
    const llama_vocab *vocab = llama_model_get_vocab(state->model);
    int n_prompt_max = strlen(prompt_str) + 128;
    std::vector<llama_token> tokens(n_prompt_max);
    int n_tokens = llama_tokenize(vocab, prompt_str, strlen(prompt_str),
                                   tokens.data(), n_prompt_max, true, true);
    env->ReleaseStringUTFChars(prompt, prompt_str);

    if (n_tokens < 0) {
        return env->NewStringUTF("[error: tokenization failed]");
    }
    tokens.resize(n_tokens);

    llama_kv_cache_clear(state->ctx);

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        llama_batch_add(batch, tokens[i], i, {0}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(state->ctx, batch) != 0) {
        llama_batch_free(batch);
        return env->NewStringUTF("[error: decode failed]");
    }

    // Generate text
    std::string output;
    int n_cur = n_tokens;

    for (int i = 0; i < max_tokens; i++) {
        llama_token new_token = llama_sampler_sample(state->sampler, state->ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        // Convert token to text
        char buf[256];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            output.append(buf, n);
        }

        llama_batch_clear(batch);
        llama_batch_add(batch, new_token, n_cur, {0}, true);
        n_cur++;

        if (llama_decode(state->ctx, batch) != 0) {
            break;
        }
    }

    llama_batch_free(batch);
    return env->NewStringUTF(output.c_str());
#else
    return env->NewStringUTF("[error: llama not available]");
#endif
}

// ============================================================
// Utility
// ============================================================

JNIEXPORT jstring JNICALL
Java_com_handy_voice_tts_QwenTTSEngine_nativeGetSystemInfo(
    JNIEnv *env, jobject thiz) {
#ifdef LLAMA_AVAILABLE
    std::string info = "llama.cpp available, ";
    info += "GGML backend: " + std::string(ggml_backend_dev_name(ggml_backend_dev_get(0)));
    return env->NewStringUTF(info.c_str());
#else
    return env->NewStringUTF("llama.cpp not available");
#endif
}

} // extern "C"

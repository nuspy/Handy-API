# Handy Voice ProGuard Rules

# Keep JNI methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep Hilt generated classes
-keep class dagger.hilt.** { *; }

# Keep ONNX Runtime
-keep class ai.onnxruntime.** { *; }

# Keep data classes used in JSON serialization
-keep class com.handy.voice.ui.VoiceProfile { *; }
-keep class com.handy.voice.ui.ChatMessage { *; }

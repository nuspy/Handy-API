use crate::{TranscriptionEngine, TranscriptionResult};
use std::path::Path;

#[derive(Debug, Clone)]
pub enum Language {
    Auto,
    Chinese,
    English,
    Japanese,
    Korean,
    Cantonese,
}

#[derive(Debug, Clone)]
pub struct SenseVoiceInferenceParams {
    pub language: Language,
    pub use_itn: bool,
}

#[derive(Debug, Clone, Default)]
pub struct SenseVoiceModelParams {
    pub quantization: Option<String>,
}

impl SenseVoiceModelParams {
    pub fn int8() -> Self {
        Self {
            quantization: Some("int8".to_string()),
        }
    }
}

pub struct SenseVoiceEngine;

impl SenseVoiceEngine {
    pub fn new() -> Self {
        Self
    }
}

impl TranscriptionEngine for SenseVoiceEngine {
    type InferenceParams = SenseVoiceInferenceParams;
    type ModelParams = SenseVoiceModelParams;

    fn load_model_with_params(
        &mut self,
        _model_path: &Path,
        _params: Self::ModelParams,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Err("SenseVoice engine is not yet implemented".into())
    }

    fn unload_model(&mut self) {}

    fn transcribe_samples(
        &mut self,
        _samples: Vec<f32>,
        _params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        Err("SenseVoice engine is not yet implemented".into())
    }
}

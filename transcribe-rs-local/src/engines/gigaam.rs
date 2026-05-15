use crate::{TranscriptionEngine, TranscriptionResult};
use std::path::Path;

#[derive(Debug, Clone, Default)]
pub struct GigaAMModelParams {}

#[derive(Debug, Clone, Default)]
pub struct GigaAMInferenceParams {}

pub struct GigaAMEngine;

impl GigaAMEngine {
    pub fn new() -> Self {
        Self
    }
}

impl TranscriptionEngine for GigaAMEngine {
    type InferenceParams = GigaAMInferenceParams;
    type ModelParams = GigaAMModelParams;

    fn load_model_with_params(
        &mut self,
        _model_path: &Path,
        _params: Self::ModelParams,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Err("GigaAM engine is not yet implemented".into())
    }

    fn unload_model(&mut self) {}

    fn transcribe_samples(
        &mut self,
        _samples: Vec<f32>,
        _params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        Err("GigaAM engine is not yet implemented".into())
    }
}

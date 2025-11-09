use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;
use candle_core::{DType, Device};

use crate::{
    models::deepseek_ocr::{config::DeepseekOCRConfig, processor::DeepseekOCRProcessor},
    tokenizer::TokenizerModel,
    utils::{get_device, get_dtype},
};

pub struct DeepseekOCRGenerateModel {
    tokenizer: TokenizerModel,
    processor: DeepseekOCRProcessor,
}

impl DeepseekOCRGenerateModel {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let tokenizer = TokenizerModel::init(path)?;
        let device = &get_device(device);
        let dtype = get_dtype(dtype, "bfloat16");
        let processor = DeepseekOCRProcessor::new(device, dtype)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: DeepseekOCRConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        
        Ok(Self {
            tokenizer,
            processor,
        })
    }

    pub fn generate(&mut self, mes: ChatCompletionParameters) -> Result<()> {
        let (input_ids, images_ori, image_crop, image_seq_mask, images_spatial_crop_t) = self
            .processor
            .process_info(&mes, &self.tokenizer, 640, 640, true)?;

        Ok(())
    }
}

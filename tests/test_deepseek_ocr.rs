use aha::models::deepseek_ocr::{generate::DeepseekOCRGenerateModel, processor::DeepseekOCRProcessor};
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};

#[test]
fn deepseek_ocr_test() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda deepseek_ocr_test -- --nocapture
    let message = r#"
    {
        "model": "deepseek-ocr",
        "messages": [
            {
                "role": "user",
                "content": [ 
                    {
                        "type": "image",
                        "image_url": 
                        {
                            "url": "file://./assets/img/ocr_test1.png"
                        }
                    },              
                    {
                        "type": "text", 
                        "text": "<image>\n<|grounding|>Convert the document to markdown. "
                    }
                ]
            },
            {
                "role": "assistant",
                "content": ""
            }
        ]
    }
    "#;
    let model_path = "/home/jhq/huggingface_model/deepseek-ai/DeepSeek-OCR/";
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let device = Device::cuda_if_available(0)?;
    let dtype = DType::BF16;
    let mut model = DeepseekOCRGenerateModel::init(model_path, Some(&device), Some(dtype))?;
    let res = model.generate(mes)?;
    Ok(())
}

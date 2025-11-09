use anyhow::{Ok, Result};
use candle_core::{IndexOp, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, Init, LayerNorm, Linear, Module, VarBuilder, conv2d, linear,
    linear_no_bias,
};

use crate::models::deepseek_ocr::config::DeepseekOCRConfig;

pub struct PatchEmbed {
    proj: Conv2d,
}

impl PatchEmbed {
    pub fn new(
        vb: VarBuilder,
        in_chans: usize,
        embed_dim: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding,
            stride,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let proj = conv2d(in_chans, embed_dim, kernel_size, cfg, vb.pp("proj"))?;
        Ok(Self { proj })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.proj.forward(xs)?;
        let xs = xs.permute((0, 2, 3, 1))?;
        Ok(xs)
    }
}

pub struct Attention {
    num_heads: usize,
    head_dim: usize,
    qkv: Linear,
    proj: Linear,
    scaling: f64,
    use_rel_pos: bool,
    rel_pos_h: Option<Tensor>,
    rel_pos_w: Option<Tensor>,
}

impl Attention {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        use_rel_pos: bool,
        input_size: Option<(usize, usize)>,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let scaling = 1.0 / (head_dim as f64).sqrt();
        let qkv = if qkv_bias {
            linear(dim, dim * 3, vb.pp("qkv"))?
        } else {
            linear_no_bias(dim, dim * 3, vb.pp("qkv"))?
        };
        let proj = linear(dim, dim, vb.pp("proj"))?;
        let mut rel_pos_h = None;
        let mut rel_pos_w = None;
        if use_rel_pos {
            if input_size.is_none() {
                return Err(anyhow::anyhow!(
                    "Input size must be provided if using relative positional encoding."
                ));
            }
            let input_size = input_size.unwrap();
            let h_len = 2 * input_size.0 - 1;
            let w_len = 2 * input_size.1 - 1;
            rel_pos_h = Some(vb.get_with_hints((h_len, head_dim), "rel_pos_h", Init::Const(0.))?);
            rel_pos_w = Some(vb.get_with_hints((w_len, head_dim), "rel_pos_w", Init::Const(0.))?);
        }

        Ok(Self {
            num_heads,
            head_dim,
            qkv,
            proj,
            scaling,
            use_rel_pos,
            rel_pos_h,
            rel_pos_w,
        })
    }

    // fn get_rel_pos(q_size: usize, k_size: usize, rel_pos: &Tensor) -> Result<Tensor> {
    //     let max_rel_dist = 2 * std::cmp::max(q_size, k_size) - 1;
    //     let rel_pos_resized = if rel_pos.dim(0)? != max_rel_dist {
    //         let dtype = rel_pos.dtype();
    //         let rel_pos = rel_pos.to_dtype(candle_core::DType::F32)?;
    //         let rel_pos_resized = 
    //     }
    // }
    // fn add_decomposed_rel_pos(&self, q: &Tensor, rel_pos_h: &Tensor, rel_pos_w: &Tensor, q_size: (usize, usize), k_size: (usize, usize)) -> Result<Tensor> {
    //     let (q_h, q_w) = q_size;
    //     let (k_h, k_w) = k_size;
        
    // }

    // pub fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
    //     let (b, h, w, _) = xs.dims4()?;
    //     // (3, B, n_head, h*w, head_dim)
    //     let qkv = self
    //         .qkv
    //         .forward(xs)?
    //         .reshape((b, h * w, 3, self.num_heads, ()))?
    //         .permute((2, 0, 3, 1, 4))?
    //         .contiguous()?;
    //     let query_states = qkv.i(0)?.contiguous()?;
    //     let key_states = qkv.i(1)?.contiguous()?;
    //     let value_states = qkv.i(2)?.contiguous()?;
    //     let xs = if self.use_rel_pos {
    //         let (rel_h, rel_w) = 
    //     } else {

    //     }
    // }
}

pub struct Block {
    norm1: LayerNorm,
    attn: Attention,
}

pub struct ImageEncoderViT {
    img_size: usize,
    patch_embed: PatchEmbed,
    pos_embed: Option<Tensor>,
    blocks: Vec<Block>,
}

pub struct VitModel {}

pub struct DeepseekV2Model {}

pub struct MlpProjector {}

pub struct DeepseekOCRModel {
    config: DeepseekOCRConfig,
    sam_model: ImageEncoderViT,
    vision_model: VitModel,
    language_model: DeepseekV2Model,
    projector: MlpProjector,
    embed_std: f64,
    image_newline: Tensor,
    view_seperator: Tensor,
    lm_head: Linear,
}

---
datasets:
- shenxq/OneVision
- shenxq/VideoChat2
base_model:
- Vision-CAIR/LongVU_Qwen2_7B_img
pipeline_tag: video-text-to-text
model-index:
- name: llava-onevision-qwen-7b-ov
  results:
  - task:
      type: multimodal
    dataset:
      name: EgoSchema
      type: egoschema
    metrics:
    - type: accuracy
      value: 67.6
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: MLVU
      type: mlvu
    metrics:
    - type: accuracy
      value: 65.4
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: MVBench
      type: mvbench
    metrics:
    - type: accuracy
      value: 66.9
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: VideoMME
      type: videomme
    metrics:
    - type: accuracy
      value: 60.6
      name: accuracy
      verified: true
---
# LongVU

This repository contains the model based on Qwen2-7B as presented in [LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding](https://huggingface.co/papers/2410.17434).

Play with the model on the [HF demo](https://huggingface.co/spaces/Vision-CAIR/LongVU).

<div align="left">
    <a href='https://vision-cair.github.io/LongVU'><img src="https://longvu.s3.amazonaws.com/assets/demo.gif" alt="Demo GIF" style="width: 100%; max-width: 650px;"></a>
</div>

# Use

We provide the simple generation process for using our model. For more details, you could refer to [Github](https://github.com/Vision-CAIR/LongVU)

```python
# git clone https://github.com/Vision-CAIR/LongVU
import numpy as np
import torch
from longvu.builder import load_pretrained_model
from longvu.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from longvu.conversation import conv_templates, SeparatorStyle
from longvu.mm_datautils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)
from decord import cpu, VideoReader

tokenizer, model, image_processor, context_len = load_pretrained_model(
    "./checkpoints/longvu_qwen", None, "cambrian_qwen",
)

model.eval()
video_path = "./examples/video1.mp4"
qs = "Describe this video in detail"

vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
fps = float(vr.get_avg_fps())
frame_indices = np.array([i for i in range(0, len(vr), round(fps),)])
video = []
for frame_index in frame_indices:
    img = vr[frame_index].asnumpy()
    video.append(img)
video = np.stack(video)
image_sizes = [video[0].shape[:2]]
video = process_images(video, image_processor, model.config)
video = [item.unsqueeze(0) for item in video]

qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
conv = conv_templates["qwen"].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=video,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0.2,
        max_new_tokens=128,
        use_cache=True,
        stopping_criteria=[stopping_criteria],
    )
pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
```

# Citation

```
@article{shen2024longvu,
    title={LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding},
    author={Shen, Xiaoqian and Xiong, Yunyang and Zhao, Changsheng and Wu, Lemeng and Chen, Jun and Zhu, Chenchen and Liu, Zechun and Xiao, Fanyi and Varadarajan, Balakrishnan and Bordes, Florian and Liu, Zhuang and Xu, Hu and J. Kim, Hyunwoo and Soran, Bilge and Krishnamoorthi, Raghuraman and Elhoseiny, Mohamed and Chandra, Vikas},
    journal={arXiv:2410.17434},
    year={2024}
  }
```
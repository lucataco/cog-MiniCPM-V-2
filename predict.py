# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from PIL import Image
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "openbmb/MiniCPM-V-2"
MODEL_CACHE = "checkpoint"
WEIGHTS_URL = "https://weights.replicate.delivery/default/openbmb/MiniCPM-V-2/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Enable hf-transfer
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        # Download with Pget
        if not os.path.exists(MODEL_CACHE):
            download_weights(WEIGHTS_URL, MODEL_CACHE)
        model = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE
        )
        # Run on T4 - fp16
        self.model = model.to(device='cuda', dtype=torch.float16)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE
        )

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Input prompt", default="What is in the image?")
    ) -> str:
        """Run a single prediction on the model"""
        img = Image.open(str(image)).convert('RGB')
        question = prompt
        msgs = [{'role': 'user', 'content': question}]
        res, context, _ = self.model.chat(
            image=img,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            sampling=True,
            temperature=0.7
        )
        return res

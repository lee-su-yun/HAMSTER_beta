import argparse
import os
from typing import List, Literal, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Import necessary components from VILA
from VILA.llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from VILA.llava.conversation import SeparatorStyle, conv_templates
from VILA.llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from VILA.llava.model.builder import load_pretrained_model
from VILA.llava.utils import disable_torch_init
from VILA.server import (
    TextContent,
    ImageURL,
    ImageContent,
    ChatMessage,
    load_image,
    normalize_image_tags,
)

# Add your custom model to the supported models
class ChatCompletionRequest(BaseModel):
    model: Literal[
        "VILA1.5-3B",
        "VILA1.5-3B-AWQ",
        "VILA1.5-3B-S2",
        "VILA1.5-3B-S2-AWQ",
        "Llama-3-VILA1.5-8B",
        "Llama-3-VILA1.5-8B-AWQ",
        "VILA1.5-13B",
        "VILA1.5-13B-AWQ",
        "VILA1.5-40B",
        "VILA1.5-40B-AWQ",
        "Hamster_dev",  # Add your custom model here
    ]
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.9
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False
    use_cache: Optional[bool] = True
    num_beams: Optional[int] = 1

# Initialize global variables
model = None
model_name = None
tokenizer = None
image_processor = None
context_len = None

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global model, model_name, tokenizer, image_processor, context_len
    disable_torch_init()
    model_path = app.args.model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_name, None
    )
    print(f"Model {model_name} loaded successfully. Context length: {context_len}")

@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        global model, tokenizer, image_processor, context_len

        if request.model != model_name:
            raise ValueError(
                f"The endpoint is configured to use the model {model_name}, "
                f"but the request model is {request.model}"
            )

        max_tokens = request.max_tokens
        temperature = request.temperature
        top_p = request.top_p
        use_cache = request.use_cache
        num_beams = request.num_beams

        messages = request.messages
        conv_mode = app.args.conv_mode

        images = []
        conv = conv_templates[conv_mode].copy()
        user_role = conv.roles[0]
        assistant_role = conv.roles[1]

        for message in messages:
            if message.role == "user":
                prompt = ""
                if isinstance(message.content, str):
                    prompt += message.content
                if isinstance(message.content, list):
                    for content in message.content:
                        if content.type == "text":
                            prompt += content.text
                        if content.type == "image_url":
                            image = load_image(content.image_url.url)
                            images.append(image)
                            prompt += IMAGE_PLACEHOLDER

                normalized_prompt = normalize_image_tags(prompt)
                conv.append_message(user_role, normalized_prompt)
            if message.role == "assistant":
                prompt = message.content
                conv.append_message(assistant_role, prompt)

        prompt_text = conv.get_prompt()
        print("Prompt input: ", prompt_text)

        if len(images) == 0:
            images_input = None
        else:
            images_tensor = process_images(images, image_processor, model.config).to(
                model.device, dtype=torch.float16
            )
            images_input = [images_tensor]

        input_ids = (
            tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(model.device)
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_input,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_tokens,
                use_cache=use_cache,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        print("\nAssistant: ", outputs)

        resp_content = [TextContent(type="text", text=outputs)]
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": request.model,
            "choices": [{"message": ChatMessage(role="assistant", content=resp_content)}],
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--workers", type=int, default=1)
    app.args = parser.parse_args()

    uvicorn.run(app, host=app.args.host, port=app.args.port, workers=app.args.workers) 
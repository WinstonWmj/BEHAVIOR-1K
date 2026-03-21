import base64
import importlib
import mimetypes
import os
from pathlib import Path
from typing import Any, Optional


DASHSCOPE_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"


def _to_image_url(image: str) -> str:
    """
    将图片输入转换成 OpenAI 兼容的 image_url。
    - 若是 http/https URL：原样返回
    - 若是本地路径：转为 data URL
    """
    if image.startswith(("http://", "https://")):
        return image

    image_path = Path(image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "application/octet-stream"

    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def describe_image_with_instruction(
    image: str,
    instruction: str,
    *,
    model: str = "qwen-vl-max-latest",
    api_key_env: str = "DASHSCOPE_API_KEY",
    timeout: Optional[float] = 60.0,
) -> str:
    """
    输入图片 + 指令，返回模型文本结果。

    参数:
    - image: 本地图片路径或公网 URL
    - instruction: 给模型的文字指令
    - model: 模型名（默认 qwen-vl-max-latest，可按需替换）
    - api_key_env: API Key 的环境变量名，默认 DASHSCOPE_API_KEY
    - timeout: 请求超时秒数
    """
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(
            f"未找到 API Key。请先设置环境变量 {api_key_env}，例如:\n"
            f"export {api_key_env}='你的key'"
        )

    try:
        openai_module = importlib.import_module("openai")
        OpenAI = getattr(openai_module, "OpenAI")
    except Exception as exc:
        raise ImportError(
            "缺少依赖 openai，请先安装: pip install openai"
        ) from exc

    client: Any = OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL, timeout=timeout)
    image_url = _to_image_url(image)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    )

    content = response.choices[0].message.content

    # 某些兼容实现可能返回字符串或分段结构，这里统一提取为纯文本。
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join(t for t in texts if t).strip()

    return str(content)


if __name__ == "__main__":
    # 示例:
    # export DASHSCOPE_API_KEY="your_key"
    # python test_aliyun.py
    import datetime
    print(datetime.datetime.now())
    result = describe_image_with_instruction(
        image="pathto/teaser.png",
        instruction="请简洁描述这张图片中的主要内容。",
        model="qwen3.5-plus",
    )
    print(result)
    print(datetime.datetime.now())

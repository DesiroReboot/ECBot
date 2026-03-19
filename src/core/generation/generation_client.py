from __future__ import annotations

import json
from time import sleep
from urllib import request

from src.config import GenerationConfig


class GenerationClient:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.base_url = str(config.base_url).rstrip("/")
        self.api_key = str(config.api_key or "").strip()
        self.model = str(config.model or "").strip()
        self.timeout = max(1, int(config.timeout))
        self.max_retries = max(0, int(config.max_retries))
        self.temperature = float(config.temperature)

    @property
    def available(self) -> bool:
        if not self.base_url or not self.model:
            return False
        if not self.api_key or self.api_key.upper().startswith("YOUR_"):
            return False
        return True

    def rewrite(
        self,
        *,
        query: str,
        template_answer: str,
        steps: list[str],
        evidence: list[str],
        citation_sources: list[str],
    ) -> str:
        if not self.available:
            raise RuntimeError("generation client unavailable")

        system_prompt = (
            "你是企业知识库问答的中文编辑器。"
            "你只能重写表达，不能新增事实、不能新增来源、不能删掉关键步骤。"
            "输出必须保持以下段落标题：问题：、建议执行步骤：、关键信息：、参考来源：。"
            "步骤必须使用 1. 2. 3. 的编号格式。"
            "若证据不足，只能更谨慎表达，不得编造。"
        )
        evidence_block = "\n".join(f"- {line}" for line in evidence) if evidence else "- 无"
        steps_block = "\n".join(f"{idx}. {line}" for idx, line in enumerate(steps, start=1))
        citations_block = "\n".join(f"- {source}" for source in citation_sources) if citation_sources else "- 无"
        user_prompt = (
            f"用户问题：{query}\n\n"
            f"模板答案：\n{template_answer}\n\n"
            f"可用步骤：\n{steps_block}\n\n"
            f"可用证据：\n{evidence_block}\n\n"
            f"可用来源：\n{citations_block}\n\n"
            "请输出最终答案。请勿包含多余说明。"
        )

        url = f"{self.base_url}/chat/completions"
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        ).encode("utf-8")
        req = request.Request(
            url=url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with request.urlopen(req, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8", errors="ignore")
                data = json.loads(body)
                content = self._extract_content(data)
                if not content:
                    raise RuntimeError("empty generation response")
                return content
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    sleep(min(1.5 * (attempt + 1), 4.0))
                    continue
                break
        raise RuntimeError(f"generation rewrite failed: {last_error}")

    def _extract_content(self, payload: dict) -> str:
        choices = payload.get("choices", [])
        if not choices:
            return ""
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message", {}) if isinstance(first, dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    text = str(block.get("text", "")).strip()
                    if text:
                        text_parts.append(text)
            return "\n".join(text_parts).strip()
        return str(content).strip()

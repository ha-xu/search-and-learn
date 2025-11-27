#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import logging
from dataclasses import dataclass

import numpy as np
from vllm import LLM, SamplingParams

logger = logging.getLogger()


def build_conv(
    prompt: str, response: str | None, system_prompt: str
) -> list[dict[str, str]]:
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    if response != "":
        conversation.append({"role": "assistant", "content": response})

    return conversation


def last(x):
    if len(x) == 0:
        logger.warning("empty list")
        return 0
    return x[-1]


def list_mean(x):
    if len(x) == 0:
        logger.warning("empty list")
        return 0
    return np.mean(x)


@dataclass
class Beam:
    prompt: str
    index: int
    current_text: str | None
    next_texts: list[str] | None
    lookahead_texts: list[str] | None
    stop_reasons: list[str | None] | None
    best_scores: list[float]  # the PRM scores
    all_scores: list[list[float]]  # all PRM scores
    previous_text: str | None
    pruned: False
    history: list[str]
    completed: bool = False
    completion_tokens: int = 0
    logprobs: list[dict] = None


@dataclass
class GenResult:
    index: int
    initial_prompt: str
    first_step_text: str
    first_step_stop_reason: str
    lookahead_text: str
    stop_reason: str | None
    total_completion_tokens: int  # modified : token count
    logprobs: list[dict]


def generate_k_steps(
    templated_convs,
    lookahead_steps: int,
    llm: LLM,
    sampling_params: SamplingParams,
    beam_width: int,
) -> list[Beam]:
    gen_results = []
    for i, text in enumerate(templated_convs):
        for j in range(beam_width):
            gen_result = GenResult(
                index=i,
                initial_prompt=text,
                first_step_text="",
                lookahead_text="",
                stop_reason=None,
                first_step_stop_reason=None,
                # modified : token count
                total_completion_tokens=0,
                logprobs=[],
            )
            gen_results.append(gen_result)

    gen_sampling_params = copy.deepcopy(sampling_params)

    for i in range(lookahead_steps + 1):
        if i == 1:
            gen_sampling_params.temperature = 0.0  # greedy for the rest of the steps
        # get all generations that did not finish with eos
        current_gen = [
            gen_results[i]
            for i in range(len(gen_results))
            if gen_results[i].stop_reason != "EOS"
        ]
        gen_prompts = [
            gen_result.initial_prompt + gen_result.lookahead_text
            for gen_result in current_gen
        ]
        llm_outputs = llm.generate(gen_prompts, gen_sampling_params, use_tqdm=False)
        for gen_result, output in zip(current_gen, llm_outputs):
            gen_text = output.outputs[0].text
            # 【修改点 B】：获取本次 LLM 调用生成的 completion_tokens
            # 假设 output 对象包含 usage 字段或类似的 token 计数属性。
            # 这里使用了常见的 LLM 输出结构：
            completion_tokens_this_step = output.outputs[0].token_ids[-1] if hasattr(output.outputs[0], 'token_ids') else len(llm.get_tokenizer().encode(gen_text, add_special_tokens=False))
            # ❗ 注意：最理想的方式是使用 output.outputs[0].completion_tokens 或 output.usage.completion_tokens
            # 请根据您使用的 LLM 框架（如 VLLM、HuggingFace 或其他）调整此处的数据获取方式。
            # 为了示例，我们使用一个假设的路径：
            completion_tokens_from_llm = output.usage.completion_tokens if hasattr(output, 'usage') else completion_tokens_this_step 

            # 【修改点 C】：累加 Token 计数到 GenResult
            gen_result.total_completion_tokens += completion_tokens_from_llm
            if i == 0:
                gen_result.first_step_text = gen_text
                gen_result.first_step_stop_reason = output.outputs[0].stop_reason
                if gen_result.first_step_stop_reason is None:
                    gen_result.first_step_stop_reason = "EOS"

            gen_result.lookahead_text = gen_result.lookahead_text + gen_text
            gen_result.stop_reason = output.outputs[0].stop_reason
            if gen_result.stop_reason is None:
                gen_result.stop_reason = "EOS"
            logprobs = output.outputs[0].logprobs
            gen_result.logprobs.append(logprobs)

    outputs: list[Beam] = []

    counter = 0
    for i, text in enumerate(templated_convs):
        next_texts = []
        stop_reasons = []
        lookahead_texts = []
        # 【修改点 D】：初始化 Beam 对象的 completion_tokens 属性
        total_tokens_for_beam = 0
        total_logprobs_for_beam = []
        for j in range(beam_width):
            gen_result = gen_results[counter]
            next_texts.append(gen_result.first_step_text)
            lookahead_texts.append(gen_result.lookahead_text)
            stop_reasons.append(gen_result.first_step_stop_reason)
            # 【修改点 E】：从 GenResult 获取最终的 Token 计数
            # 确保这里只添加一次，因为 beam_width 内的 GenResult 属于同一个 prompt/conv
            if j == 0: # 只从第一个 GenResult 提取本次生成的总 Token 数
                 total_tokens_for_beam = gen_result.total_completion_tokens
            counter += 1
            total_logprobs_for_beam.extend(gen_result.logprobs)

        beam_result = Beam(
            prompt=text,
            index=i,
            current_text="",
            next_texts=next_texts,
            lookahead_texts=lookahead_texts,
            stop_reasons=stop_reasons,
            best_scores=[0.0],
            all_scores=[],
            previous_text=None,
            pruned=False,
            history=[],
            # 【修改点 F】：将 Token 计数添加到 Beam 结果中
            completion_tokens=total_tokens_for_beam,
            logprobs=total_logprobs_for_beam,
        )
        outputs.append(beam_result)

    return outputs

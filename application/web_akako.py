import os, sys
import random
import threading

import gradio as gr
import mdtex2html

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)

from arguments import ModelArguments, DataTrainingArguments

model = None
tokenizer = None

"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


stop_flag = False


def predict(input, chatbot, max_length, top_p, temperature, history):
    global stop_flag
    chatbot.append((parse_text(input), ""))
    for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        chatbot[-1] = (parse_text(input), parse_text(response))

        if stop_flag:
            stop_flag = False
            raise gr.Info("强制性响应中止")

        yield chatbot, history


def stop():
    global stop_flag
    stop_flag = True


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">❤️‍🔥 Chat Akako - 内测版 ❤️‍🔥</h1>
    """)
    gr.Markdown("""  
当前加载模型：`Akako-int8-4.0Msamples\checkpoint-17850`

1. 如果跟我聊天的人数过多，您的请求可能不会立即响应，请您理解 😊
2. 我有时会胡言乱语，包括但不限于咒骂、侮辱、讽刺、脏话，请您原谅 🙏
3. 如果我一直在重复，单击“停止响应”按钮以强制中断我的对话 🫢
4. 如果你想要清除我的记忆，单击“清除历史”按钮 🗑️
5. 您可以调整创造力和热情值使我达到不同的对话效果 😎
    """)

    welcome = ['欢迎来和我聊天！🤗', '想要聊些什么吗！☺️', '你好呀！想聊点什么呢？😉']

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder=random.choice(welcome), lines=5).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                with gr.Row():
                    submitBtn = gr.Button("📤️发送消息", variant="primary")
                    stopBtn = gr.Button("⏹️停止响应", variant="stop")
                    emptyBtn = gr.Button("🗑️清除历史")
        with gr.Column(scale=1):
            # max_length = gr.Slider(0, 4096, value=512, step=1.0, label="最大长度", interactive=False)
            max_length = gr.Slider(0, 4096, value=512, step=1.0, label="最大长度", interactive=False)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="创造力", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="热情值", interactive=True)

    history = gr.State([])
    stopBtn.click(stop, queue=False)
    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)


def main():
    global model, tokenizer

    parser = HfArgumentParser((
        ModelArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        model_args = parser.parse_args_into_dataclasses()[0]

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)

    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection

    if model_args.ptuning_checkpoint is not None:
        print(f"Loading prefix_encoder weight from {model_args.ptuning_checkpoint}")
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
        prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    else:
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)

    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)

    if model_args.pre_seq_len is not None:
        # P-tuning v2
        model = model.half().cuda()
        model.transformer.prefix_encoder.float().cuda()

    model = model.eval()
    demo.queue().launch(share=True, inbrowser=True)


if __name__ == "__main__":
    main()

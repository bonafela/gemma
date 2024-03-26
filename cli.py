# Copyright (c) 2024 Bonafela AI - bonafela.co.za
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

import ast
import gc
import os
import socket
import subprocess
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import transformers
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles import merge_styles, Style
from prompt_toolkit.styles.pygments import style_from_pygments_cls
from pygments.lexers.shell import BashLexer
from pygments.styles import get_style_by_name
from rich.console import Console
from rich.markdown import Markdown
from transformers import AutoTokenizer, Pipeline

def get_network_ip() -> str:
    """Returns the IP address of the network interface.
    :return: IP address or localhost
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.connect(('<broadcast>', 0))
        return s.getsockname()[0]
    except OSError:
        return "localhost"


def set_settings(key_value: dict, args: Namespace, console: Console) -> None:
    """Sets the model inference and application parameters based on the key-value dictionary
    and updates Namespace object.
    :param key_value: dictionary of key-value pairs for the model and training parameters.
    Example key_value = {"markdown": True, "max_new_tokens": 1024, "temperature": 0.2}
    :param args: Namespace object containing the model inference and application parameters.
    :param console: High-level console interface.
    :returns: None
    """
    for key, value in key_value.items():
        match key:
            case "markdown":
                args.markdown = bool(value)
            case "max_new_tokens":
                args.max_new_tokens = int(value)
            case "temperature":
                temperature = float(value)
                if 0 < temperature <= 1:
                    args.temperature = temperature
            case "top_k":
                args.top_k = float(value)
            case "top_p":
                top_p = float(value)
                if 0 < top_p <= 1:
                    args.top_p = top_p
                args.top_p = float(value)
            case "prompt_output":
                args.prompt_output = str(value)
            case "prompt_input":
                args.prompt_input = str(value)
            case _:
                console.print(f'>>> ignoring unknown variable {key}', style="misty_rose1")



def show_settings(args: Namespace) -> None:
    """Prints the model inference and application parameters.
    :param args: Namespace object containing the model inference and application parameters.
    :returns: None
    """
    key_value = {"markdown": args.markdown,
                 "max_new_tokens": args.max_new_tokens,
                 "top_k": args.top_k,
                 "top_p": args.top_p,
                 "prompt_output": args.prompt_output,
                 "prompt_input": args.prompt_input,
                 }
    for key, value in key_value.items():
        print(key.ljust(16) + ":", str(value))
    print("\n")


def process_settings(command: str | None, args: Namespace, console: Console):
    """Display or set model inference and application parameters based on the key-value dictionary.
    :param command: dictionary of key-value pairs for the model and training parameters. Example key_value =
    {"markdown": True, "max_new_tokens": 1024, "temperature": 0.2}
    :param args: Namespace object containing the model inference and application parameters.
    :param console: High-level console interface.
    :returns: None
    """
    if command is None:
        return False
    if not command.strip():
        return False
    if command.startswith("settings="):
        settings = command.split("=", 1)[1]
        try:
            key_value = ast.literal_eval(settings)
        except (ValueError, SyntaxError):
            console.print('>>> settings is invalid dictionary', style="misty_rose1")
            return True
        if type(key_value) is not dict:
            console.print('>>> settings is invalid dictionary', style="misty_rose1")
            return True
        set_settings(key_value, args, console)
        return True
    if command.startswith("settings") and len(command.split()) == 1:
        show_settings(args)
        return True
    return False


def process_command(command: str | None, console: Console, pipeline: Pipeline, args: Namespace,
                    is_cli: bool = True) -> float:
    """Processes a command from the user and generates text using the specified model.

    :param command: The user's command.
    :param console: High-level console interface.
    :param pipeline: Transformers pipeline object.
    :param args: Namespace object containing the model inference and application parameters.
    :param is_cli: Whether the script is being run in the command line interface.
    :returns: time to process a prompt command
    """
    execution = 0.0

    if command is None or len(command.split()) <= 1:
        return execution
    process = subprocess.run(command, shell=True, capture_output=True)
    if process.stderr:
        console.print(">>> Your OS did not like that command\n", style="misty_rose1")
        if not is_cli:
            print(command)
        return execution
    msg_user_content = process.stdout.decode('utf-8')
    if not msg_user_content.strip():
        return execution

    processing_msg = "Processing "
    if not is_cli:
        processing_msg += args.prompt_output
    start = time.time()
    with console.status(processing_msg, spinner="bouncingBall"):
        messages = [
            {"role": "user", "content": f"{msg_user_content}"},
        ]
        prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipeline(
            prompt,
            max_new_tokens=args.max_new_tokens,
            add_special_tokens=True,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            pad_token_id=pipeline.tokenizer.eos_token_id,
            prompt_lookup_num_tokens=10, # https://github.com/apoorvumang/prompt-lookup-decoding
        )
    end = time.time()
    execution = end - start
    results = outputs[0]["generated_text"][len(prompt):]

    if is_cli:
        print("\n")
        if args.markdown:
            md = Markdown(results)
            console.print(md, width=120, justify="full")
        else:
            print(f"{results}")

    if args.prompt_output:
        console_msg = ""
        Path(args.prompt_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.prompt_output, "w") as f:
            f.write(results)
        if is_cli:
            console_msg += f"This output is also written in"
        console.print(f"\n>>> {console_msg} {args.prompt_output}", style="grey70")

    gc.collect()
    torch.cuda.empty_cache()
    return execution


def main_batch(console, pipeline, args) -> None:
    """The main function that handles batch processing of prompts.

    This function reads a file containing a list of prompts, processes each prompt, and generates text using the
    specified model.

    :param console: High-level console interface.
    :param pipeline: Transformers pipeline object.
    :param args: Namespace object containing the model inference and application parameters.
    :returns: None
    """
    with open(args.prompt_input) as f:
        for command in f.readlines():
            if process_settings(command, args, console):
                continue
            process_command(command, console, pipeline, args, is_cli=False)
    # restore defaults
    args.prompt_input = None
    args.prompt_output = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.prompt_output.md')


def main_cli(console: Console, pipeline: Pipeline, args: Namespace) -> None:
    """The main function that handles user interaction and model inference.
    :param console: high level console interface.
    :param args: Namespace object containing the model inference and application parameters.
    :param pipeline: transformers Pipeline object
    :returns: None
    """
    execution = 0.0

    def toolbar_callable():
        toolbar = "\n\n\n"
        toolbar += f"markdown={args.markdown} "
        toolbar += f"| max_new_tokens={args.max_new_tokens} "
        toolbar += f"| temperature={args.temperature} "
        toolbar += f"| top_k={args.top_k} "
        toolbar += f"| top_p={args.top_p} "
        toolbar += f"| execution={execution:0.2f}s "
        toolbar += f"| Ctr-D=quit "
        return toolbar

    message = [
        ('class:model', f'{args.model}'),
        ('class:at', '@'),
        ('class:host', f'{get_network_ip()}'),
        ('class:message', ' How can I assist you?'),
        ('class:pound', '\n$ '),
    ]
    style = merge_styles([
        style_from_pygments_cls(get_style_by_name(args.prompt_style)),
        Style.from_dict({
            'model': '#ffff7f',
            'at': '#ffff7f',
            'host': '#ffff7f',
            'message': '#c9d8e7',
            'pound': '#c1a0ff',
            "bottom-toolbar": "fg:lightblue bg:default noreverse"
        })
    ])
    session = PromptSession(message=message, lexer=PygmentsLexer(BashLexer), style=style,
                            include_default_pygments_style=False, history=FileHistory(args.prompt_history),
                            auto_suggest=AutoSuggestFromHistory())
    while True:
        try:
            command = session.prompt(bottom_toolbar=toolbar_callable)
            if process_settings(command, args, console):
                if args.prompt_input:
                    if os.path.isfile(args.prompt_input):
                        main_batch(console, pipeline, args)
                    else:
                        console.print(f">>> prompt_input={args.prompt_input} is not a valid file\n",
                                      style="misty_rose1")
                continue
            execution = process_command(command, console, pipeline, args)
            continue
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
    print('>>> Good Bye!')


def parse_args() -> Namespace:
    """Parses arguments for the script.
    :returns: Namespace object containing the model inference and application parameters.
    """
    parser = ArgumentParser(description="Generate text from prompts in interactive or batch mode")

    parser.add_argument("--model",
                        default='google/gemma-7b-it',
                        help="google gemma model")

    parser.add_argument("--quant",
                        action='store_true',
                        help="load model in 4-bit quantization")

    # the following can be modified during runtime
    parser.add_argument("--markdown",
                        action='store_true',
                        help="pretty print markdown")

    parser.add_argument("--prompt_history",
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), '.prompt_history'),
                        help="file to store prompt entries")

    parser.add_argument("--prompt_style",
                        default='github-dark',
                        help="prompt syntax highlight style")

    parser.add_argument("--prompt_output",
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), '.prompt_output.md'),
                        help="file to store prompt output")

    parser.add_argument("--prompt_input",
                        type=str,
                        help="file containing prompt inputs")

    parser.add_argument("--max_new_tokens",
                        default=1024,
                        help="max new tokens")

    parser.add_argument("--temperature",
                        default=0.2,
                        help="temperature")

    parser.add_argument("--top_k",
                        default=50, help="top k")

    parser.add_argument("--top_p",
                        default=0.7, help="top p")

    return parser.parse_args()


def main(args: Namespace) -> None:
    """The main function that handles user interaction and model inference.
    :param args: Namespace object containing the model inference and application parameters.
    :returns: None
    """
    pipeline = transformers.pipeline(
        "text-generation",
        tokenizer=AutoTokenizer.from_pretrained(args.model),
        model=args.model,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "quantization_config": {"load_in_4bit": True} if args.quant else None,
        },
        device_map="auto",
    )

    Path(args.prompt_history).parent.mkdir(parents=True, exist_ok=True)
    Path(args.prompt_history).touch(exist_ok=True)
    console = Console()

    if args.prompt_input and os.path.isfile(args.prompt_input):
        main_batch(console, pipeline, args)
    else:
        main_cli(console, pipeline, args)


if __name__ == '__main__':
    parser_args = parse_args()
    main(parser_args)

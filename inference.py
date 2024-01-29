import os

DEVICE_INDEX = 1
DEVICE_NUM = 3
inf_size = int(1800/DEVICE_NUM)
inf_start = DEVICE_INDEX*inf_size
inf_end = (DEVICE_INDEX+1)*inf_size
import sys
import json
import re
import fire
import torch
from tqdm import tqdm
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from filtering.bert_model import RM_model
import numpy as np

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "minlik/chinese-llama-7b-merged",
    lora_weights: str = "./models",
    prompt_template_file: str = "./templates/zeroshot/zeroshot.json",
    output_save_dir: str = f'./inference/results',
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
#-------*******************************-------
    if os.path.exists(output_save_dir)==False:
        os.mkdir(output_save_dir)

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        local_files_only=True,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.
        model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        prompt=None,
        temperature=1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
    ):

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output
    
    def clean(text):
        text = text.strip()
        if '\n' not in text:
            return text
        else:
            return ' '.join(text.split('\n'))

    with open('./summary/test/test_text.txt', 'r', encoding='UTF-8') as f:
        lines = f.readlines()

    cur_files = next(os.walk(output_save_dir))[-1]
    cur_idx_list = [int(file.split(".")[0]) for file in cur_files]
    for idx in tqdm(range(inf_start,inf_end)):
        if idx in cur_idx_list:
            continue
        line = lines[idx]
        text = line.strip().split("###")[0]
        name = line.strip().split("###")[1]
        import time
        with open(prompt_template_file, 'r') as f:
            prompt_template = json.load(f)
        
        prompt = prompt_template['cot_template'].format(name=name,text=text)


        output = evaluate(prompt=prompt)
        output = re.sub("[<s>]","",output)
        
        with open(f'{output_save_dir}/{idx}.txt', 'a+', encoding='UTF-8') as f:
            f.write(output)

    
if __name__ == "__main__":
    fire.Fire(main)
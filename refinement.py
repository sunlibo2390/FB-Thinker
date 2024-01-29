import os
import sys
import json
import re
import fire
import torch
from tqdm import tqdm
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
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
    lora_weights: str = "./models/",
    data_for_improve: str = "./reward_model/data/reward_results.json",
    output_save_dir: str = f'./refinemnet/results',
    hard_templates_path: str = "./templates/improve/inference/hard_templates.json",
    soft_template_path: str = "./templates/improve/inference/soft_template.txt",
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
        # local_files_only=True,
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
        max_new_tokens=512,
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
    
    sys.path.append('./filtering')

    with open(data_for_improve, 'r', encoding='UTF-8') as f:
        item_list = json.load(f)

    cur_files = next(os.walk(output_save_dir))[-1]
    cur_idx_list = [int(file.split(".")[0]) for file in cur_files]
    for idx in tqdm(range(len(item_list))):
        if idx in cur_idx_list:
            continue
        data_point = item_list[idx]

        text = data_point['text']
        summary = data_point['summary']
        name = data_point['name']
        label0 = data_point['fact_label']
        label1 = data_point['compre_label']
        label2 = data_point['relate_label']

        with open(hard_templates_path,'r') as f:
            hard_templates = json.load(f)
        # if label2==1:
        #     prompt = hard_templates['relate_template'].format(
        #         text=text, summary=summary, name=name
        #     )
        # else:
        #     prompt = hard_templates['empty_template'].format(
        #         text=text, summary=summary, name=name
        #     )
        if label0==1 and label1==0 and label2==0:
            prompt = hard_templates['fact_template'].format(
                text=text, summary=summary, name=name
            )
        if label0==0 and label1==1 and label2==0:
            prompt = hard_templates['compre_template'].format(
                text=text, summary=summary, name=name
            )
        if label0==0 and label1==0 and label2==1:
            prompt = hard_templates['relate_template'].format(
                text=text, summary=summary, name=name
            )
        if label0==1 and label1==1 and label2==0:
            prompt = hard_templates['fact_compre_template'].format(
                text=text, summary=summary, name=name
            )
        if label0==0 and label1==1 and label2==1:
            prompt = hard_templates['compre_relate_template'].format(
                text=text, summary=summary, name=name
            )
        if label0==1 and label1==0 and label2==1:
            prompt = hard_templates['relate_fact_template'].format(
                text=text, summary=summary, name=name
            )
        if label0==1 and label1==1 and label2==1:
            prompt = hard_templates['tri_template'].format(
                text=text, summary=summary, name=name
            )
        if label0==0 and label1==0 and label2==0:
            prompt = hard_templates['empty_template'].format(
                text=text, summary=summary, name=name
            )
            # with open(soft_template_path,'r') as f:
            #     soft_template = f.read()
            # prob0 = data_point['fact_score']
            # prob1 = data_point['compre_score']
            # prob2 = data_point['relate_score']
            # prompt = soft_template.format(
            #     text=text, summary=summary, name=name, prob0=prob0*100, prob1=prob1*100, prob2=prob2*100,
            # )

        output = evaluate(prompt=prompt)
        output = re.sub("[</s>]","",output)

        print("\n\nOUTPUT\n\n"+output)
        
        with open(f'{output_save_dir}/{idx}.txt', 'a+', encoding='UTF-8') as f:
            f.write(output)

    
if __name__ == "__main__":
    fire.Fire(main)
from pathlib import Path
import re
import pandas as pd
from vllm import LLM, SamplingParams

LLM_MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct' # 'Qwen/Qwen2-7B-Instruct'

def generate_templated_column(df, template):
    # Find all placeholders in the template
    placeholders = re.findall(r'{(.*?)}', template)
    
    # Function to replace placeholders with actual column values for each row
    def apply_template(row):
        filled_template = template
        for placeholder in placeholders:
            if placeholder in row:
                filled_template = filled_template.replace(f'{{{placeholder}}}', str(row[placeholder]))
        return filled_template
    
    # Apply the function to each row and create a new column with the templated text
    return df.apply(apply_template, axis=1)
    
class BaseConfig:
    def __init__(self, model_name, max_new_tokens):
        self.model_name = model_name
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        self.llm = None
    
    def create_llm(self, force_create=False):
        if self.llm is None or force_create or self.llm.llm_engine.get_model_config().hf_config.name_or_path != self.model_name:
            self.llm = LLM(model=self.model_name)
            
    def generate_batch(self, prompt, df, resp_col = 'llm_resp'):
        """
        vllm apply to dataframe (inference prompt over all rows)
        prompt - string with {col_name_1} ... {col_name_2} place holders --> df[col_name_xxx] are inserted into place holders.
          ) If the instructions are already embedded in a col called 'text' - simply pass prompt='{text}' text_col (diff prompt each row)
        """
        generating_prompts = generate_templated_column(df, prompt).tolist() 
        
        # The llm.generate call will batch all prompts and send the batch at once
        # if resources allow. The prefix will only be cached after the first batch
        # is processed, so we need to call generate once to calculate the prefix
        # and cache it.
        outputs = self.llm.generate(generating_prompts[0], self.sampling_params)

        if len(df) > 1:
            # Subsequent batches can leverage the cached prefix
            outputs = self.llm.generate(generating_prompts, self.sampling_params)

        # Print the outputs. You should see the same outputs as before
        gen_texts = []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            gen_texts.append(generated_text)
            #print(f"Generated text: {generated_text!r}")
            
        df[resp_col] = gen_texts   


class TaskConfig(BaseConfig):
    def __init__(self, model_name, max_new_tokens):
        super().__init__(model_name, max_new_tokens)
        
        


class MainConfig:
    def __init__(self):
        self.summary = TaskConfig(model_name = LLM_MODEL_NAME, max_new_tokens = 2000)        
        self.summary.verify = TaskConfig(model_name = LLM_MODEL_NAME, max_new_tokens = 2000)

cfg = MainConfig()        
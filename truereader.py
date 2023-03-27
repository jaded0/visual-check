import transformers
from transformers import LlamaForCausalLM, AutoTokenizer
from typing import Optional, List, Union, Tuple, Any, Dict, Sequence
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
import torchvision.models as models
from dataclasses import dataclass, field
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from typing import Callable
from transformers import Trainer
import io
import subprocess as sub
from pdf2image import convert_from_bytes, convert_from_path
from PIL import Image
import tempfile
import re


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

class ImageEmbeddingModel(torch.nn.Module):
    def __init__(self):
        super(ImageEmbeddingModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1])) # Remove last layer

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1) # Flatten
        return x

# conver latex to image, temporary file approach, supports batches.
from typing import List

def latex_to_images_tempfile(latex_strs: List[str], dpi: int = 150) -> List[Image.Image]:
    images = []

    for latex_str in latex_strs:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save the LaTeX string to a temporary .tex file
            tex_path = os.path.join(tmpdir, "temp.tex")
            with open(tex_path, "w") as tex_file:
                tex_file.write(latex_str)

            # Run pdflatex on the .tex file to generate the PDF
            pdf_path = os.path.join(tmpdir, "temp.pdf")
            sub.run(["pdflatex", "-output-directory", tmpdir, tex_path], check=True)

            # Convert the PDF to an image using pdf2image
            img = convert_from_path(pdf_path, dpi=dpi, fmt="png")[0]

            images.append(img)

    return images


# convert latex to image in memory
def latex_to_image(latex_str: str, dpi: int = 150) -> Image.Image:
    # Compile the LaTeX code into a PDF using pdflatex
    latex_process = sub.run(
        ["pdflatex", "-halt-on-error", "-jobname", "temp", "-output-format", "pdf", "-"],
        input=latex_str,
        stdout=sub.DEVNULL,
        stderr=sub.PIPE,
        check=True,
        # capture_output=True,
        text=True,
        shell=True
    )


    if latex_process.returncode != 0:
        raise RuntimeError(f"pdflatex failed: {latex_process.stderr}")

    # Read the generated PDF into a BytesIO object
    pdf_data = io.BytesIO(latex_process.stdout)
    print(pdf_data.getvalue())
    # Run pdftoppm on the PDF data to generate the image
    pdftoppm_process = sub.run(
        ["pdftoppm", "-png", "-rx", "150", "-ry", "150", "-", "-"],
        input=pdf_data.getvalue(),
        capture_output=True,
        check=True,
    )


    if pdftoppm_process.returncode != 0:
        raise RuntimeError(f"pdftoppm failed: {pdftoppm_process.stderr}")

    # Read the generated image into a BytesIO object
    image_data = io.BytesIO(pdftoppm_process.stdout)

    # Open the image using PIL
    image = Image.open(image_data)

    return image

class TrueReader(LlamaForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embed_image = ImageEmbeddingModel()
        self.embeddings_project = torch.nn.Linear(self.config.hidden_size + 2048, self.config.hidden_size)

    # def is_valid_latex(self, latex_code: str) -> bool:
    #     # You can use a more sophisticated check, but for now, we'll use a simple regex pattern
    #     pattern = r"\\documentclass.*?\\begin{document}.*?\\end{document}"
    #     return bool(re.search(pattern, latex_code, re.DOTALL))

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            images: torch.FloatTensor = None,
        ) -> Union[Tuple, CausalLMOutputWithPast]:

        print(f'in forward pass. input_ids size before embedding: {input_ids.shape}')
        text_embeddings = self.model.embed_tokens(input_ids)
    
        # Get the image embeddings using ResNet
        image_embeddings = self.embed_image(images)
        
        # check if the shape of both embeddings
        print(f'In forward pass. text embeddings: {text_embeddings.shape}, image embeddings: {image_embeddings.shape}')

        # Resize the image embeddings to match the text embeddings shape
        image_embeddings = image_embeddings.view(image_embeddings.size(0), 1, -1).repeat(1, text_embeddings.size(1), 1)
        print(f'resized image embeddings: {image_embeddings.shape}')
        # Concatenate the text and image embeddings
        combined_embeddings = torch.cat([text_embeddings, image_embeddings], dim=-1)
        inputs_embeds = self.embeddings_project(combined_embeddings)

        # Run through transformer
        output = super().forward(input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        return output
        # compute loss
        # Generate LaTeX code and render it as a PNG image
        # token_ids = output.logits.argmax(dim=-1).squeeze(0)
        # print(f'in forward function, token id shape: {token_ids.shape}')
        # latex_code = tokenizer.batch_decode(token_ids.tolist())  # Convert the model's output to LaTeX code
        # print(f'latex code: {latex_code}')
        # try:
        #     gen_image = latex_to_images_tempfile(latex_code)   # Render the LaTeX code as a PNG image
        # except RuntimeError as e: # if the latex code does not compile, return inf loss
        #     print(f'Runtime error: {e}, \nlatex code: {latex_code} does not compile')
        #     return CausalLMOutputWithPast(
        #         loss=torch.inf,
        #         logits=output.logits,
        #         past_key_values=output.past_key_values,
        #         hidden_states=output.hidden_states,
        #         attentions=output.attentions,
        #     )

        
        # kl_div_loss = self.compute_kl_div_loss(images, gen_image)
        # loss = kl_div_loss

        # return CausalLMOutputWithPast(
        #     loss=loss,
        #     logits=output.logits,
        #     past_key_values=output.past_key_values,
        #     hidden_states=output.hidden_states,
        #     attentions=output.attentions,
        # )
    
    # def compute_kl_div_loss(self, ref_image: torch.Tensor, gen_image: torch.Tensor) -> torch.Tensor:
    #     ref_image_prob = F.softmax(ref_image.view(ref_image.size(0), -1), dim=-1)
    #     gen_image_prob = F.softmax(gen_image.view(gen_image.size(0), -1), dim=-1)
    #     kl_div_loss = F.kl_div(torch.log(gen_image_prob), ref_image_prob, reduction="batchmean")
    #     return kl_div_loss

    # def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None, hiddens: Any = None) -> torch.Tensor:
    #     images = batch.pop("images")
    #     outputs = self(**batch)
    #     lm_loss = outputs.loss
        
    #     # Generate LaTeX code and render it as a PNG image
    #     latex_code = tokenizer.decode(outputs.logits.argmax(dim=-1).squeeze(0))  # Convert the model's output to LaTeX code
    #     gen_image = latex_to_image_tempfile(latex_code)   # Render the LaTeX code as a PNG image
        
    #     kl_div_loss = self.compute_kl_div_loss(images, gen_image)
        
    #     loss = lm_loss + kl_div_loss
    #     return loss
from transformers import Trainer

def generate_latex_code(model, input_ids, tokenizer, images, max_length=100):
    batch_size = input_ids.size(0)
    gen_tokens = input_ids
    eos_token_id = tokenizer.eos_token_id
    past_key_values = None
    
    for _ in range(max_length):
        outputs = model(input_ids=gen_tokens, past_key_values=past_key_values, images=images)
        logits = outputs.logits[:, -1, :]
        next_tokens = torch.argmax(logits, dim=-1).unsqueeze(-1)
        gen_tokens = torch.cat((gen_tokens, next_tokens), dim=1)

        # Stop generating tokens if EOS token is reached for all sequences
        if torch.all(next_tokens.view(-1) == eos_token_id):
            break
        
        past_key_values = outputs.past_key_values

    latex_code_list = [tokenizer.decode(token_ids) for token_ids in gen_tokens]
    return latex_code_list



class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # You can leave this empty since you'll calculate the loss in the training_step method

        if return_outputs:
            return None, inputs
        return None
    
    def compute_kl_div_loss(self, ref_image: torch.Tensor, gen_image: torch.Tensor) -> torch.Tensor:
        ref_image_prob = F.softmax(ref_image.view(ref_image.size(0), -1), dim=-1)
        gen_image_prob = F.softmax(gen_image.view(gen_image.size(0), -1), dim=-1)
        kl_div_loss = F.kl_div(torch.log(gen_image_prob), ref_image_prob, reduction="batchmean")
        return kl_div_loss
    
    def training_step(self, model, inputs):
        images, input_ids = inputs["images"], inputs["input_ids"]
        images = images.to(self.args.device)
        input_ids = input_ids.to(self.args.device)

        # Generate full LaTeX sequence
        gen_latex_code = generate_latex_code(model, input_ids, self.tokenizer, images, max_length=100)

        # Calculate custom loss
        loss = 0
        for latex_code, original_image in zip(gen_latex_code, images):
            try:
                gen_image = latex_to_images_tempfile([latex_code])   # Render the LaTeX code as a PNG image
            except RuntimeError as e: # if the latex code does not compile, return inf loss
                print(f'Runtime error: {e}, \nlatex code: {latex_code} does not compile')
                return {"loss": torch.inf} # TODO: check if inf loss is correct
            
            gen_image_tensor = transforms.to_tensor(gen_image).to(self.args.device)  # Convert generated image to tensor

            # Compute KL divergence loss between original image and generated image
            kl_loss = self.compute_kl_div_loss(original_image, gen_image_tensor)
            loss += kl_loss

        loss = loss / len(images)

        return {"loss": loss}

class ImageTextDataset(Dataset):
    def __init__(self, img_dir, prompt):
        super(ImageTextDataset, self).__init__()
        self.img_dir = img_dir
        self.prompt = prompt
        self.img_files = os.listdir(img_dir)
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, i) -> dict:
        img_path = os.path.join(self.img_dir, self.img_files[i])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        text = self.prompt
        input_ids = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)["input_ids"].squeeze(0)
        # print(f"Dataset item text: {input_ids}")
        print(f"in dataloader. input shape: {input_ids.shape}, image shape: {image.shape}")
        return { "input_ids": input_ids, "images": image}

class ImageTextDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # print(f"Batch: {batch}")
        images = [item["images"] for item in batch]
        input_ids = [item["input_ids"] for item in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # input_ids = self.tokenizer(input_ids, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        print(f'in collator. input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}, images shape: {torch.stack(images).shape}')
        return {"input_ids": input_ids, "attention_mask": attention_mask, "images": torch.stack(images)}

def make_image_text_data_module(tokenizer, img_dir, prompt):
    dataset = ImageTextDataset(img_dir, prompt)
    data_collator = ImageTextDataCollator(tokenizer)
    return {"train_dataset": dataset, "eval_dataset": None, "data_collator": data_collator}



if (__name__ == "__main__"):
    # Example usage
    latex_str = r"""
    \documentclass{article}
    \begin{document}
    Hello, world!
    \end{document}
    """
    print(latex_str)
    # image = latex_to_image(latex_str)
    # image.show()
    print('now trying with tempfiles')
    image_tempfile = latex_to_images_tempfile([latex_str])
    # image_tempfile.show()
    # image_tempfile.save('image_tempfile_test.png')
    print('done with tempfiles')

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = TrueReader.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    img_dir = "pdfs/images/"
    prompt = "recreate the image in latex."
    data_module = make_image_text_data_module(tokenizer, img_dir, prompt)

    trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()

     # Save the trainer state
    trainer.save_state()

    # Save the model using the safe_save_model_for_hf_trainer function
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
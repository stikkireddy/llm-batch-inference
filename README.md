# fast-batch-inference

This is a WYSIWYG (what you see is what you get).

It is to help you do batch inference on GPUs in your vnet/vpc. It takes advantage of using vLLM 
which has custom optimizations for throughput for batch inference. We run this on a Databricks single node
or cluster.

There are no real configurations for this, just make sure you use the right vm type for this to work.

Any model the size of llama 70b - Mixtral 8x7b make sure to use atleast 2 A100s. Everything else can use A100 or smaller. Will post a better table of model size and number of gpus needed to host one instance.

The reason to use vLLM is it supports batching ootb so it will be rare to hit OOM errors for passing in a larger payload as it will figure out how much space there is available and batch appropriately. It will though throw OOM if you dont have enough memory to load the model as well as room for KV cache.

The plan is to have 3 notebooks.

1. Batch scoring for single node (multi or single gpu). [DONE]
2. Batch scoring for multi node (multi or single gpu). [TBD]
3. Batch scoring by making api calls to provisioned throughput models hosted on model serving. [TBD]


## Prompting & Performance

Most  OSS models have specific instruction tokens and special tokens to deal with prompting and sending instructions to the model. This is **extremely** important for throughput and performance. Otherwise the model will be very chatty and potentially loop completions till the max token has been met. This is where these special tokens come into play.

For now mixtral and llama based models use similar tokens and work similar to xml/html tags with subtle differences explained below:

1. `[INST]` and `[/INST]` to indicate instruction blocks
2. `<<SYS>>` and `<</SYS>>` to indicate system prompt
3. `<s>` and `</s>` to indicate beginning of string (BOS) and end of string (EOS) respectively

### LLAMA Models

The Llama2 series requires the use of `<<SYS>>` for creaing systen prompts and using `[INST]` tokens for giving specific instructions. 
please note `<s>` is not closed

Prompt Template: 
```
<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_message }} [/INST]
```
Example:
```
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

There's a llama in my garden 😱 What should I do? [/INST]
```
Further details [here]( https://huggingface.co/blog/llama2#how-to-prompt-llama-2 )

### Mixtral and Mistral models

Both these family of models only use the `<s>` and `[INST]` tokens for creating the prompt structure.

Prompt Template: 
```
<s>[INST] Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]
```
Example:
```
[INST] You are a helpful code assistant. Your task is to generate a valid JSON object based on the given information:
name: John
lastname: Smith
address: #1 Samuel St.
Just generate the JSON object without explanations:
[/INST]
```
Further details [here]( https://www.promptingguide.ai/models/mixtral#prompt-engineering-guide-for-mixtral-8x7b )


## Inference Sampling Parameters

When the model is trying to predict the next token the following parameters impact the accuracy / consistency of the results.

1. Temperature controls randomness: Lower values make responses more deterministic, higher values increase diversity. (Typically ranges from 0 - 1)

2. P (top-p) controls the probability mass: Lower values focus on more likely tokens, cutting off the less likely ones.


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

Most models have specific instruction tokens and special tokens to deal with prompting and sending instructions to the model. This is **extremely** important for throughput and performance. Otherwise the model will be very chatty and potentially loop completions till the max token has been met. This is where these special tokens come into play.

For now mixtral and llama based models use the following instructions and the work similar to xml/html tags which require a starting and enclosing tag:

1. `[INST]` and `[/INST]` to indicate instruction blocks
2. `<<SYS>>` and `<</SYS>>` to indicate system prompt
3. `<s>` and `</s>` to indicate beginning of string (BOS) and end of string (EOS) respectively

Example:

This takes advantage of the above mentioned tokens.

```
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

There's a llama in my garden 😱 What should I do? [/INST]
```

Please keep in mind that both this and the output tokens configured during inference impact decoding performance and this can drastically improve throughput if your outputs need to be consise and short.


## Inference Sampling Parameters

When the model is trying to predict the next token the following parameters impact the accuracy / consistency of the results.

1. Temperature controls randomness: Lower values make responses more deterministic, higher values increase diversity. (Typically ranges from 0 - 1)

2. P (top-p) controls the probability mass: Lower values focus on more likely tokens, cutting off the less likely ones.


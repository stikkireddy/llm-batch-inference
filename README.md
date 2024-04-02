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


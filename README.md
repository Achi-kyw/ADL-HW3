# ADL_HW3_RAG

## usage
- install package (python 3.12) 
<pre><code>pip install -r requirements.txt</code></pre>

- train retriever

<pre><code>python train_retriever.py --output_dir ./output_retriever --batch_size 32</code></pre>

- train reranker

<pre><code>python train_reranker.py --output_dir ./output_reranker --batch_size 32</code></pre>

- save corpus into vector database (from corpus.txt)
<pre><code>python save_embbedings.py --retriever_model_path [your_model_path] --build_db</code></pre>

- create `./.env` and place your own hf_token([link](https://huggingface.co/docs/hub/security-tokens)) into `hf_token="....."`
- inference
<pre><code>python inference_batch.py --test_data_path [your_data_path] --retriever_model_path [your_retrieve_model_path] --reranker_model_path [your_rerank_model_path] --test_data_path ./data/test_open.txt</code></pre>

CUDA_VISIBLE_DEVICES=1 python train_retriever.py --output_dir ./output_retriever10 --batch_size 32
CUDA_VISIBLE_DEVICES=2 python train_reranker.py --output_dir ./output_reranker10 --batch_size 32


CUDA_VISIBLE_DEVICES=3 python inference_batch.py --test_data_path ./data/test_open.txt --retriever_model_path intfloat/multilingual-e5-small --reranker_model_path ./output_reranker9 --result_file_name output_infbaseretriever >output_infbaseretriever.txt 2>&1

CUDA_VISIBLE_DEVICES=1 python save_embeddings.py --retriever_model_path ./output_retriever9  --build_db --ouputfolder vector_test

CUDA_VISIBLE_DEVICES=1 python inference_batch.py --test_data_path ./data/test_small.txt --retriever_model_path ./output_retriever9 --reranker_model_path ./output_reranker9 --result_file_name output_infpromptthink >output_infpromptthink.txt 2>&1


CUDA_VISIBLE_DEVICES=1 python inference_batch.py --test_data_path ./data/test_open.txt --retriever_model_path intfloat/multilingual-e5-small --reranker_model_path cross-encoder/ms-marco-MiniLM-L-12-v2 --result_file_name output_base --index_folder ./vector_test >output_infbaseall.txt 2>&1

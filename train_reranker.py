import json
import os
import random
import torch
from typing import List, Dict
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse

def load_corpus(path: str) -> Dict[str, str]:
    corpus = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            pid, p_text, title = data.get("id"), data.get("text"), data.get("title")
            if pid and p_text is not None:
                full_text = (title.strip() + "\n" if title else "") + p_text.strip()
                if full_text: corpus[pid] = full_text
    print(f"Loaded {len(corpus)} passages.")
    return corpus

def load_qrels(path: str) -> Dict[str, str]:
    qrels = {}
    with open(path, 'r', encoding='utf-8') as f:
        qrels_data = json.load(f)
        for qid, pid_dict in qrels_data.items():
            qrels[qid] = list(pid_dict.keys())[0]
    print(f"Loaded {len(qrels)} q-p relations.")
    return qrels

def build_reranker_examples(train_txt_path: str, qrels_map: Dict[str, str], corpus_map: Dict[str, str]) -> List[InputExample]:
    examples = []
    all_passage_ids = list(corpus_map.keys())
    if not all_passage_ids: raise ValueError("Corpus is empty.")

    with open(train_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            qid, query_text = data.get('qid'), data.get('rewrite')
            if not (qid and query_text and qid in qrels_map): continue
            
            positive_passage_id = qrels_map[qid]
            positive_passage_text = corpus_map.get(positive_passage_id)
            if positive_passage_text is None: continue

            examples.append(InputExample(texts=[query_text, positive_passage_text], label=1.0))

            negative_passage_id = positive_passage_id
            while negative_passage_id == positive_passage_id:
                negative_passage_id = random.choice(all_passage_ids)
            
            negative_passage_text = corpus_map.get(negative_passage_id)
            if negative_passage_text:
                examples.append(InputExample(texts=[query_text, negative_passage_text], label=0.0))

    print(f"Total InputExamples created: {len(examples)}")
    return examples

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    corpus_map = load_corpus(args.corpus)
    qrels_map = load_qrels(args.qrels)
    examples = build_reranker_examples(args.train_txt, qrels_map, corpus_map)

    model = CrossEncoder(args.model_name, num_labels=1, device=device)
    dataloader = DataLoader(examples, shuffle=True, batch_size=args.batch_size, collate_fn=lambda batch: batch)
    loss_fct = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_record = []
    global_step = 0
    print(f"Starting training on {device}...")
    model.train()

    for epoch in range(args.epochs):
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=100)
        for batch in progress:
            batch_texts = [example.texts for example in batch]
            batch_labels = [example.label for example in batch]
            
            features = model.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
            features = {k: v.to(device) for k, v in features.items()}
            batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.float).to(device)

            optimizer.zero_grad()
            logits = model(**features).logits
            loss = loss_fct(logits.squeeze(-1), batch_labels_tensor)
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            progress.set_postfix(loss=f"{current_loss:.4f}")
            global_step += 1
            if global_step % args.loss_checkpoint == 0:
                loss_record.append(current_loss)

    print(f"Training finished. Saving model to {args.output_dir}")
    model.save(args.output_dir)
    if loss_record:
        csv_path = os.path.join(args.output_dir, "loss_record.csv")
        with open(csv_path, "w") as f:
            f.write("step,loss\n")
            for i, l in enumerate(loss_record):
                step_num = (i + 1) * args.loss_checkpoint
                f.write(f"{step_num},{l}\n")
        print(f"Loss record saved to {csv_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train a Sentence Transformer Reranker")
    ap.add_argument("--train_txt", type=str, default="./data/train.txt")
    ap.add_argument("--qrels", type=str, default="./data/qrels.txt")
    ap.add_argument("--corpus", type=str, default="./data/corpus.txt")
    ap.add_argument("--output_dir", type=str, default="./output_reranker")
    ap.add_argument("--model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    ap.add_argument("--batch_size", type=int, default=16) # Reranker  batch size 建議較小
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--loss_checkpoint", type=int, default=10, help="Record loss every N steps")
    args = ap.parse_args()
    main(args)
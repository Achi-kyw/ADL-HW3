import json
import os
import random
import torch
from typing import List, Dict
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse

def load_corpus(path: str) -> Dict[str, str]:
    corpus = {}
    print(f"Loading corpus from: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            pid = data.get("id")
            p_text = data.get("text")
            title = data.get("title")
            if pid and p_text is not None:
                full_text = ""
                if title: full_text += title.strip() + "\n"
                full_text += p_text.strip()
                if full_text: corpus[pid] = full_text
    print(f"Loaded {len(corpus)} passages.")
    return corpus

def load_qrels(path: str) -> Dict[str, str]:
    qrels = {}
    print(f"Loading qrels from: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        qrels_data = json.load(f)
        for qid, pid_dict in qrels_data.items():
            pos_pid = list(pid_dict.keys())[0]
            qrels[qid] = pos_pid
    print(f"Loaded {len(qrels)} query->positive passage relations.")
    return qrels

def build_mnrl_examples(train_txt_path: str, qrels_map: Dict[str, str], corpus_map: Dict[str, str]) -> List[InputExample]:
    examples = []
    print(f"Building training examples from: {train_txt_path}")
    with open(train_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            qid = data.get('qid')
            query_text = data.get('rewrite')
            if not qid or query_text is None: continue

            positive_passage_id = qrels_map.get(qid)
            if not positive_passage_id: continue

            positive_passage_text = corpus_map.get(positive_passage_id)
            if positive_passage_text is None: continue

            examples.append(InputExample(texts=[f"query: {query_text}", f"passage: {positive_passage_text}"]))

    print(f"Total InputExamples created: {len(examples)}")
    return examples

def main(train_txt, qrels, corpus, output_dir, model_name="intfloat/multilingual-e5-small", batch_size=32, epochs=1, lr=1e-5, loss_checkpoint=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)

    corpus_map = load_corpus(corpus)
    qrels_map = load_qrels(qrels)
    if not corpus_map or not qrels_map: return
    examples = build_mnrl_examples(train_txt, qrels_map, corpus_map)
    if not examples: return

    print(f"Loading base model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    dataloader = DataLoader(
        examples,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True,
        collate_fn=model.smart_batching_collate
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    loss_record = []
    global_step = 0

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_total_loss = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)

        for batch in progress:
            features, labels = batch

            features_on_device = []
            for feature_dict in features:
                 features_on_device.append({k: v.to(device) for k, v in feature_dict.items()})
            if labels is not None:
                labels = labels.to(device)

            optimizer.zero_grad()
            loss = train_loss(features_on_device, labels)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            epoch_total_loss += current_loss
            progress.set_postfix(loss=f"{current_loss:.4f}")
            global_step += 1

            if global_step % loss_checkpoint == 0:
                 loss_record.append(current_loss)

        avg_epoch_loss = epoch_total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} finished. Average Epoch Loss: {avg_epoch_loss:.4f}")

    print(f"Training finished. Saving final model to {output_dir}")
    model.save(output_dir)
    print(f"Model saved successfully to {output_dir}")

    if loss_record:

        csv_path = os.path.join(output_dir, "loss_record.csv")
        with open(csv_path, "w") as f:
            f.write("step,loss\n")
            for i, l in enumerate(loss_record):
                f.write(f"{(i+1)*loss_checkpoint},{l}\n")
        print(f"Loss record saved to {csv_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train a Sentence Transformer Retriever")
    ap.add_argument("--train_txt", type=str, default="./data/train.txt")
    ap.add_argument("--qrels", type=str, default="./data/qrels.txt")
    ap.add_argument("--corpus", type=str, default="./data/corpus.txt")
    ap.add_argument("--output_dir", type=str, default="./output_retriever")
    ap.add_argument("--model_name", type=str, default="intfloat/multilingual-e5-small")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)

    args = ap.parse_args()
    main(**vars(args))
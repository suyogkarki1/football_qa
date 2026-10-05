# Football QA

A small question-answering model for football trivia, built with PyTorch. It reads a question like *"Who won the 1954 FIFA World Cup?"* and picks the answer (`WestGermany`) from the set of known answers.

## Dataset

`football_qa_dataset_full.csv` has 2,882 question/answer pairs covering FIFA World Cups, clubs, players and transfers.

- 839 unique answers, all single tokens (multi-word names are hyphenated, e.g. `Real-Madrid`, `Inter-Miami`)
- Most facts are asked in a few rephrased ways, e.g. "Who won…", "Which team was crowned champion of…"

## Models

There are two notebooks, trained and tested on the same split so their results can be compared.

### 1. LSTM from scratch: `rnn.ipynb`


1. **Tokenize**: lowercase, strip punctuation (hyphens kept)
2. **Split**: 80% train+validation / 20% test, with 15% of the training part held out for validation
3. **Vocab**: built from training questions only; unseen words map to `<UNK>`
4. **Batching**: questions are padded to equal length (batch size 32) and packed so the LSTM ignores padding
5. **Model**: embedding (128) → bidirectional LSTM (128) → dropout (0.4) → linear layer over the 839 answers
6. **Training**: Adam (lr 1e-3, weight decay 1e-5), early stopping when validation accuracy stops improving for 10 epochs
7. **Predict**: `predict(model, "question")` prints the answer, or `idk` if confidence is below 0.3

### 2. Pretrained sentence embeddings: `embeddings.ipynb`

1. Each question is turned into a 768-dimensional vector by the pretrained [`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) model, which already knows that phrases like "won" and "was crowned champion" mean similar things
2. A logistic regression classifier is trained on those vectors to pick the answer, with its regularisation strength chosen on the validation set

## Results

| Set | LSTM (`rnn.ipynb`) | Embeddings (`embeddings.ipynb`) |
|---|---|---|
| Train | 73.0% | 97.3% |
| Validation | 19.9% | 26.9% |
| Test | 14.6% | **20.5%** |

The embeddings model handles reworded questions better. For example, it answers *"team lost the final of 1954 FIFA World Cup?"* with `Hungary` (correct), where the LSTM says `idk`.

Test accuracy is still low for both, because of the dataset rather than the models:

- About 19% of test answers never appear in the training data, so they can't be predicted.
- 482 of the 839 answers occur only once in the whole dataset.
- Many questions differ by only a word or two (World Cup winner vs. host vs. runner-up for the same year), and each specific fact appears only a few times.

For comparison, a nearest-neighbour baseline (copy the answer of the most similar training question) gets 10.8%. Adding more paraphrases of each question would help most.

## Running it

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install torch pandas jupyter sentence-transformers scikit-learn
jupyter notebook
```

Open either notebook and run all cells. Each takes a few minutes on CPU. The first run of `embeddings.ipynb` downloads the ~400 MB pretrained model.

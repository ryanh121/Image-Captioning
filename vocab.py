import nltk
import pickle
import numpy as np
from collections import Counter
from pycocotools.coco import COCO

captions_path = '/projects/training/bawc/IC/annotations/captions_train2014.json'
idx2word_path = 'idx2word'
word2idx_path = 'word2idx'

coco = COCO(captions_path)
counter = Counter()
ids = coco.anns.keys()
for i, id in enumerate(ids):
    caption = str(coco.anns[id]['caption'])
    tokens = nltk.tokenize.word_tokenize(caption.lower())
    counter.update(tokens)

    if (i+1) % 1000 == 0:
        print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

len(counter)
counts = list(counter.values())
counts.sort(reverse=True)
hist = np.histogram(counts, bins=[1, 10, 100, 1000, 10000])
print(hist)
hist[0][1:].sum()

threshold = hist[0][1:].sum()

sum(counts[:threshold])/sum(counts)

# If the word frequency is less than 'threshold', then the word is discarded.
vocab = ['<pad>', '<unk>', '<start>', '<end>']
vocab.extend([word for word, count in sorted(
    list(counter.items()), key=lambda x: x[1], reverse=True)[:threshold]])

word2idx = {}
idx2word = {}
for i, word in enumerate(vocab):
    word2idx[word] = i
    idx2word[i] = word


mysum = 0
for word in idx2word.values():
    mysum += counter[word]

print(mysum/sum(counts))

with open(idx2word_path, 'wb') as f:
    pickle.dump(idx2word, f)

with open(word2idx_path, 'wb') as f:
    pickle.dump(word2idx, f)

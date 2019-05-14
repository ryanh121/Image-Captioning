import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import torch.utils.data as data
# from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision
import torchvision.transforms as transforms
from utils import save_checkpoint
import torch.nn.functional as f
from pycocotools.coco import COCO
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import nltk
from PIL import Image


class BLEU4Dataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: word2idx dictionary
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.img_ids = list(self.coco.imgToAnns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        img_id = self.img_ids[index]
        captions = []
        for anns in coco.imgToAnns[img_id]:
            temp = anns['caption']
            tokens = nltk.tokenize.word_tokenize(str(temp).lower())
            caption = []
            caption.append(vocab['<start>'])
            caption.extend([vocab.get(token, 1) for token in tokens])
            caption.append(vocab['<end>'])
            captions.append(caption)

        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, captions

    def __len__(self):
        return len(self.img_ids)


def collate_fn_BLEU4(data):
    """Creates mini-batch tensors from the list of tuples (image, captions).

    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        captions: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    images, batch_captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images)

    batch_captions = list(batch_captions)
    return images, batch_captions

def get_loader_BLEU4(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = BLEU4Dataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn_BLEU4)
    return data_loader



#num_classes = 100
model_name = 'dropoutandlayer8'
start_model_idx = 15
end_model_idx = 19
batch_size = 256
embed_size = 512
hidden_size = 512
learning_rate = 0.0001
gradient_clip = 5
EDKidx = 3
#DIM = 224

test_image_dir = '/projects/training/bawc/IC/val2014/'
test_caption_path = '/projects/training/bawc/IC/annotations/captions_val2014.json'
vocab_path = 'word2idx'

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform_BLEU4 = transforms.Compose([
    transforms.Resize((224, 224), interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load word2idx dictionary
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

BLEU4loader = get_loader_BLEU4(test_image_dir, test_caption_path, vocab,
                        transform_BLEU4, batch_size, shuffle=False, num_workers=8)
#print('finished BLEU4loader')
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers=1)
encoder = encoder.to(device)
decoder = decoder.to(device)
#params = list(decoder.parameters()) + list(encoder.linear.parameters())
#optimizer = torch.optim.Adam(params, lr=learning_rate)
print('finished model initialization')
checkpoints = os.listdir('checkpoint')

total_step = len(BLEU4loader)
for model_idx in range(start_model_idx,end_model_idx+1):

    #encoder = EncoderCNN(embed_size)
    #decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers=1)
    #encoder = encoder.to(device)
    #decoder = decoder.to(device)
    #params = list(decoder.parameters()) + list(encoder.linear.parameters())
    #optimizer = torch.optim.Adam(params, lr=learning_rate)

    #BLEU4loader = get_loader_BLEU4(test_image_dir, test_caption_path, vocab,
    #transform_BLEU4, batch_size, shuffle=True, num_workers=8)
    #print('finished BLUE4loader')

    if checkpoints:
        for cp in checkpoints:
            name, num = cp[:-4].split('_')
            num = int(num)
            if name == model_name and model_idx == num:
                state_dict = torch.load(
                    'checkpoint/{}_{}.tar'.format(model_name, num))
                encoder.load_state_dict(state_dict['encoder_state_dict'])
                decoder.load_state_dict(state_dict['decoder_state_dict'])
                #optimizer.load_state_dict(state_dict['optimizer_state_dict'])
                print('model_{}_{} is being used'.format(name,state_dict['epoch']))
                break 

    # test
    decoder.eval()
    encoder.eval()

    with torch.no_grad():
        all_ref = []
        all_pred = []
        #print('to device finish')
        for i, (images, batch_captions) in enumerate(BLEU4loader):
            if i >= 40:
                continue
            all_ref.extend(batch_captions)
            images = images.to(device)
            #all_ref.extend(batch_captions)
            
            # Generate an caption from the image
            feature = encoder(images)
            all_pred.extend(decoder.beam_search(feature))
            #for a_pred in decoder.beam_search(feature):
            #    print(a_pred[-1] == EDKidx)
            #    if a_pred[-1] == EDKidx:
            #        all_pred.append(a_pred[1:-1])
            #    else:
            #        all_pred.append(a_pred[1:])
            #print('append finished')
            #print('pred: {}'.format(all_pred[-1]))
            #print('true: {}'.format(all_ref[-1]))
            if (i+1) % 10 == 0:
                print('model_idx {}, Step [{}/{}], shape: {}, {}'.format(
                    model_idx, i+1, total_step, 
                    (len(all_ref),len(all_ref[0]),len(all_ref[0][0])),
                    (len(all_pred),len(all_pred[0]))))
        #print(type(all_ref[0]),type(all_pred),type(all_pred[0])) 
        with open('./{}_BLEU4'.format(model_name), 'a') as f:
            print('model_idx {}, BLEU4: {:.4f}'.format(
                model_idx, corpus_bleu(all_ref,all_pred)),file=f)
        print('finished a model')

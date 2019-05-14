import torch
import numpy as np 
import pickle 
import os
from torchvision import transforms 
from model import EncoderCNN, DecoderRNN

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from pycocotools.coco import COCO

print('finished loading module')

model_name = 'dropoutandlayer8'
start_model_idx = 0
end_model_idx = 14
idx2word_path = 'idx2word'
embed_size = 512
hidden_size = 512

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

coco = COCO('/projects/training/bawc/IC/annotations/captions_val2014.json')

img_ids = [203564, 179765, 322141, 16977]
img_paths = []
for img_id in img_ids:
    img_paths.append('/projects/training/bawc/IC/val2014/' + coco.loadImgs(img_id)[0]['file_name'])
    # img_paths.append('val2014/' + coco.loadImgs(img_id)[0]['file_name'])

def load_images(image_paths, transform=None):
    
    images = []
    original_images = []
    for image_path in image_paths:
        original_images.append(Image.open(image_path))
        image = original_images[-1].convert('RGB')
        if transform is not None:
            image = transform(image)
        images.append(image)
    images = torch.stack(images)
    return images, original_images

def plot(samples):
    num = int(np.sqrt(len(img_ids)))
    fig = plt.figure(figsize=(num*5, num*5), dpi = 300)
    gs = gridspec.GridSpec(num, num)
    #gs.update(wspace=0.02, hspace=0.02)

    for i, (image, pred_caption) in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        #ax.set_aspect('equal')
        plt.title(pred_caption, fontsize = 8)
        plt.imshow(image)
    return fig


transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load idx2word
with open(idx2word_path, 'rb') as f:
    idx2word = pickle.load(f)

encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, len(idx2word), num_layers=1)
encoder = encoder.to(device)
decoder = decoder.to(device)

checkpoints = os.listdir('checkpoint')

for model_idx in range(start_model_idx,end_model_idx+1):

    if checkpoints:
        for cp in checkpoints:
            name, num = cp[:-4].split('_')
            num = int(num)
            if name == model_name and model_idx == num:
                state_dict = torch.load(
                    'checkpoint/{}_{}.tar'.format(model_name, num))
                encoder.load_state_dict(state_dict['encoder_state_dict'])
                decoder.load_state_dict(state_dict['decoder_state_dict'])
                break

    # test
    decoder.eval()
    encoder.eval()

    with torch.no_grad():
        # Prepare an image
        images, original_images = load_images(img_paths, transform)
        images = images.to(device)
        
        # Generate an caption from the image
        feature = encoder(images)
        print('Encoder finished')
        pred_ids = decoder.beam_search(feature)
        print('beam search finished')

        # Convert word_ids to words
        pred_captions = []
        for pred_id in pred_ids:
            temp = []
            for word_id in pred_id:
                temp.append(idx2word[word_id])
                if temp[-1] == '<end>':
                    #pred_captions.append(' '.join(temp))
                    break
            if len(temp) > 8:
                temp[len(temp)//2] = temp[len(temp)//2] + '\n'
            pred_captions.append(' '.join(temp))
    print('finished caption generation')
    print(pred_captions)
    print(images.size(),len(pred_captions))
    result = zip(original_images,pred_captions)
    fig = plot(result)
    plt.savefig('{}_{}_NIC'.format(model_name,model_idx),bbox_inches='tight')
    plt.close(fig)


# result = zip(original_images,['1','2','3','4'])
# fig = plot(result)
# plt.savefig('samplefig',bbox_inches='tight',dpi=400)
# plt.close(fig)


import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
# from build_vocab import Vocabulary
# from model import EncoderCNN, DecoderRNN
from dropoutmodel import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision
import torchvision.transforms as transforms
from utils import save_checkpoint
import torch.nn.functional as f

#num_classes = 100
model_name = 'dropoutandlayer8'
num_epochs = 20
batch_size = 128
embed_size = 512
hidden_size = 512
learning_rate = 0.00001
gradient_clip = 5
#DIM = 224

train_image_dir = '/projects/training/bawc/IC/train2014/'
test_image_dir = '/projects/training/bawc/IC/val2014/'
train_caption_path = '/projects/training/bawc/IC/annotations/captions_train2014.json'
test_caption_path = '/projects/training/bawc/IC/annotations/captions_val2014.json'
vocab_path = 'word2idx'

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform_train = transforms.Compose([
    transforms.Resize((224, 224), interpolation=2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224), interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load word2idx dictionary
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

# data_loader = get_loader(args.image_dir, args.caption_path, vocab,
#                              transform, args.batch_size,
#                              shuffle=True, num_workers=args.num_workers)

trainloader = get_loader(train_image_dir, train_caption_path,
                         vocab, transform_train, batch_size, shuffle=True, num_workers=8)

testloader = get_loader(test_image_dir, test_caption_path, vocab,
                        transform_test, batch_size, shuffle=False, num_workers=8)

checkpoints = os.listdir('checkpoint')

encoder = EncoderCNN(embed_size, layer8=True)
decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers=1, dropout=0.2)
encoder = encoder.to(device)
decoder = decoder.to(device)
params = list(decoder.parameters()) + list(encoder.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)
cur_epoch = 0

if checkpoints:
    num_checkpoint = -1
    for cp in checkpoints:
        name, num = cp[:-4].split('_')
        num = int(num)
        if name == model_name and num_checkpoint < num:
            num_checkpoint = num
    if num_checkpoint > -1:
        state_dict = torch.load(
            'checkpoint/{}_{}.tar'.format(model_name, num_checkpoint))
        encoder.load_state_dict(state_dict['encoder_state_dict'])
        decoder.load_state_dict(state_dict['decoder_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        cur_epoch = state_dict['epoch'] + 1
    #else:
    #    state_dict = torch.load(
    #        'checkpoint/{}_{}.tar'.format('baseline', 10))
    #    encoder.load_state_dict(state_dict['encoder_state_dict'])
    #    decoder.load_state_dict(state_dict['decoder_state_dict'])
    #    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    #    cur_epoch = state_dict['epoch'] + 1


# encoder = encoder.to(device)
# decoder = decoder.to(device)
criterion = nn.CrossEntropyLoss()

total_step = len(trainloader)
for epoch in range(cur_epoch, num_epochs):

    decoder.train()
    encoder.train()

    # for group in optimizer.param_groups:
    #     for p in group['params']:
    #         state = optimizer.state[p]
    #         if('step' in state and state['step'] >= 1024):
    #             state['step'] = 1000

    trainloss = 0

    for i, (images, captions, lengths) in enumerate(trainloader):

        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)
        lengths -= 1
        targets = pack_padded_sequence(captions[:,1:], lengths, batch_first=True)[0]

        # Forward, backward and optimize
        features = encoder(images)
        outputs = decoder(features, captions[:,:-1], lengths)
        loss = criterion(outputs, targets)
        trainloss += loss
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()

        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.clamp_(-gradient_clip, gradient_clip)

        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step'] >= 1024):
                    state['step'] = 1000

        optimizer.step()

        # Print log info
        # if i % args.log_step == 0:
        #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
        #           .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch +
                                                                     1, num_epochs, i+1, total_step, loss.item()))
            # with open('./{}_train_loss'.format(model_name), 'a') as f:
            #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()), file = f)
    trainloss /= i
    # Save the model checkpoints
    save_checkpoint(model_name, epoch, encoder, decoder, optimizer)

    # test
    decoder.eval()
    encoder.eval()

    with torch.no_grad():
        testloss = 0
        for i, (images, captions, lengths) in enumerate(testloader):
            images = images.to(device)
            captions = captions.to(device)
            lengths -= 1
            targets = pack_padded_sequence(
                captions[:,1:], lengths, batch_first=True)[0]

            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions[:,:-1], lengths, train = False)
            loss = criterion(outputs, targets)
            testloss += loss
        testloss /= i
    
    with open('./{}_loss'.format(model_name), 'a') as f:
        print('Epoch [{}/{}], trainloss: {:.4f}, testloss: {:.4f}'.format(epoch +
                                                                          1, num_epochs, trainloss.item(), testloss.item()), file=f)
    
    print('saved loss')


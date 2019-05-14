import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as f
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        # self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        # features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        #self.softmax = nn.Softmax(dim=1)
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        _, state = self.lstm(features.unsqueeze(1))
        embeddings = self.embed(captions)
        # embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed, state)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def gready_search(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

    def beam_search(self,features, states=None, beam_size=3, STKidx=2, EDKidx=3):
        """Generate captions for given image features using beam search."""
        batch_size = features.size(0)

        inputs = features.unsqueeze(1)
        _, states = self.lstm(inputs, states)

        results = []

        for image_idx in range(batch_size):
            state = (states[0][:,[image_idx],:],states[1][:,[image_idx],:])

            STK_embed = self.embed(torch.tensor(STKidx).cuda()).view(1, 1, -1)

            hidden, state = self.lstm(STK_embed,state)        # hidden: (1,1,hidden_size)

            log_prob = f.log_softmax(self.linear(hidden.squeeze(1)),dim=1)  # log_prob: (1,vocb_size)
            topk_log_prob, topk_idx = log_prob.topk(beam_size,1)        # (1,beam_size)

            topk_log_score = topk_log_prob.view(-1,1)
            topk_seq = [[STKidx, idx] for idx in topk_idx.view(-1).tolist()]
            state = (state[0].repeat(1,beam_size,1),state[1].repeat(1,beam_size,1))   # (1,beam_size,hidden_size)
            input = self.embed(topk_idx.view(-1).cuda()).view(beam_size,1,-1)   # (beam_size,1,embed_size)

            for cur_length in range(self.max_seg_length):

                hidden, state = self.lstm(input,state)                      # state[0]: (1,beam_size,hidden_size)
                log_prob = f.log_softmax(self.linear(hidden.squeeze(1)),dim=1)    # (beam_size,vocb_size)
                topk_log_prob, topk_idx = log_prob.topk(beam_size,1)              # (beam_size,beam_size)

                for beam_idx in range(beam_size):
                    if topk_seq[beam_idx][-1] == EDKidx:
                        topk_log_prob[beam_idx][0] = 0.0
                        topk_idx[beam_idx][0] = EDKidx

                topk_log_score, flattened_indices = (topk_log_score + topk_log_prob).view(-1).topk(beam_size)  # (beam_size,)
                topk_log_score = topk_log_score.view(-1,1)
                rows, cols = np.unravel_index(flattened_indices,topk_idx.size())

                temp_seq = []
                for i,j in zip(rows,cols):
                    temp_seq.append(topk_seq[i]+[topk_idx[i,j].item()])
                topk_seq  = temp_seq

                finish_counts = 0
                for seq in topk_seq:
                    if seq[-1] == EDKidx:
                        finish_counts += 1
                
                if finish_counts == beam_size:
                    break

                #topk_idx = topk_idx.view(-1).topk(beam_size)[0]
                topk_idx = topk_idx.view(-1)[flattened_indices]
                state = (state[0][:, rows, :],state[1][:, rows, :])   # (1,beam_size,hidden_size)
                input = self.embed(topk_idx.cuda()).view(beam_size,1,-1)   # (beam_size,1,embed_size)

            results.append(topk_seq[topk_log_score.view(-1).argmax().item()])
        return results

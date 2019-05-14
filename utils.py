import pickle
import torch

def save_checkpoint(model_name, epoch, encoder, decoder, optimizer):
    path = 'checkpoint/{}_{}.tar'.format(model_name,epoch)
    torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, path)

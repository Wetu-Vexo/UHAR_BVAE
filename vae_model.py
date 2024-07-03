import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,INPUT_SIZE,HIDDEN_SIZE,NUM_LAYERS,DROP_END):
        super(Encoder,self).__init__()

        self.input_size = INPUT_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.num_layers = NUM_LAYERS
        self.dropout = DROP_END
        self.bidirectional = True

        self.encoder_rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)#UNRELEASED!
    def forward(self, x):
        out, hidden = self.encoder_rnn(x)
        hidden = torch.cat((hidden[0,:,:],hidden[1,:,:],hidden[2,:,:],hidden[3,:,:]),dim=1)
        return hidden 
    
class Lambda(nn.Module):
    def __init__(self,ZDIM, hidden_size):
        super(Lambda,self).__init__()

        self.hidden_size = hidden_size*4
        self.latent_len = ZDIM

        self.hid2mean = nn.Linear(self.hidden_size,self.latent_len)
        self.hid2var = nn.Linear(self.hidden_size,self.latent_len)

    def forward(self,hidden):
        self.mean = self.hid2mean(hidden)
        self.logvar = self.hid2var(hidden)

        std = torch.exp(0.5*self.logvar)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(self.mean), self.mean, self.logvar
    
class Decoder(nn.Module):
    def __init__(self, HIDDEN_SIZE, OUTPUT_SIZE, LATENT_SIZE, NUM_LAYERS, DROP_DEC):
        super(Decoder, self).__init__()
        self.hidden_size = HIDDEN_SIZE
        self.output_size = OUTPUT_SIZE
        self.latent_size = LATENT_SIZE
        self.num_layers = 1
        self.drop_dec = DROP_DEC

        self.rnn_rec = nn.GRU(self.latent_size, self.hidden_size, self.num_layers, bias=True,
                              batch_first=True, dropout=self.drop_dec, bidirectional=True)
        self.hidden_fac = 2 * self.num_layers if self.rnn_rec.bidirectional else self.num_layers

        self.latent_hid = nn.Linear(self.latent_size, self.hidden_size * self.hidden_fac)
        self.output_layer = nn.Linear(self.hidden_size * 2 if self.rnn_rec.bidirectional else self.hidden_size, self.output_size)
    def forward(self, x, z):
        batch_size, seq_len, _ = x.size()
        
        # Prepare the initial hidden state from latent vector
        z = self.latent_hid(z).view(self.num_layers * 2, batch_size, self.hidden_size)  # Correct the reshaping based on your architecture

        # Decoder RNN takes the initial state from latent space
        dec_out, _ = self.rnn_rec(x, z)

        # Pass the output sequence through the output layer
        output_seq = self.output_layer(dec_out)
        return output_seq
class Decoder_Future(nn.Module):
    def __init__(self,TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_STEPS, hidden_size_pred, dropout_pred):
        super(Decoder_Future,self).__init__()
        
        self.num_features = NUM_FEATURES
        self.future_steps = FUTURE_STEPS
        self.sequence_length = TEMPORAL_WINDOW
        self.hidden_size = hidden_size_pred
        self.latent_length = ZDIMS
        self.n_layers  = 1
        self.dropout   = dropout_pred
        self.bidirectional = True
        
        self.rnn_pred = nn.GRU(self.latent_length, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)
        
        self.hidden_factor = (2 if self.bidirectional else 1) * self.n_layers # NEW
        
        self.latent_to_hidden = nn.Linear(self.latent_length,self.hidden_size * self.hidden_factor)
        self.hidden_to_output = nn.Linear(self.hidden_size*2, self.num_features)
        
    def forward(self, inputs, z):
        batch_size = inputs.size(0)
        
        hidden = self.latent_to_hidden(z)
        hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        
        inputs = inputs[:,:self.future_steps,:]
        decoder_output, _ = self.rnn_pred(inputs, hidden)
        
        prediction = self.hidden_to_output(decoder_output)
         
        return prediction


    
class VAutoencoder(nn.Module):
    def __init__(self,INPUT_SIZE,HIDDEN_SIZE,LATENT_SIZE,NUM_LAYERS,seq_len,DROP_ENC,DROP_DEC):
        super(VAutoencoder,self).__init__()

        self.seq_len = seq_len
        self.encoder = Encoder(INPUT_SIZE,HIDDEN_SIZE,NUM_LAYERS,DROP_ENC)
        self.latent_lam = Lambda(LATENT_SIZE,HIDDEN_SIZE)
        self.decoder = Decoder(HIDDEN_SIZE,INPUT_SIZE,LATENT_SIZE,NUM_LAYERS,DROP_DEC)
    
    def forward(self,seq):
        hidd = self.encoder(seq)
        z,mean_z,logvar_z = self.latent_lam(hidd)
        ins = z.unsqueeze(2).repeat(1,1,self.seq_len)
        ins = ins.permute(0,2,1)
        out = self.decoder(ins,z)

        return out, z ,mean_z,logvar_z

from transformers import DistilBertTokenizer, DistilBertModel, AutoModel, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased').to(device)

reuters = load_dataset("reuters21578","ModApte")
reuters.reset_format()

def tokenize(batch):
    return tokenizer(batch['text'], padding= True, truncation=True)
  
reuters_encoded =  reuters.map(tokenize, batched=True, batch_size=None)

def forward_pass(batch):
    input_ids = torch.tensor(batch['input_ids']).to(device)
    attention_mask = torch.tensor(batch['attention_mask']).to(device)
    with torch.no_grad():
        last_hidden_state = model(input_ids, attention_mask).last_hidden_state
        last_hidden_state = last_hidden_state.cpu().numpy()
    # Using Average of Unmasked hidden states for classification
    lhs_shape = last_hidden_state.shape
    bool_mask = ~np.array(batch['attention_mask']).astype(bool)
    bool_mask = np.repeat(bool_mask, lhs_shape[-1], axis=-1)
    bool_mask = bool_mask.reshape(lhs_shape)
    masked_mean = np.ma.array(last_hidden_state, mask= bool_mask).mean(axis=1)
    batch['hidden_state'] = masked_mean.data
    return batch
# Gerenating Feature Layer
reuters_encoded = reuters_encoded.map(forward_pass, batched=True, batch_size = 16)

bert_emb_xtrain = np.array(reuters_encoded['train']['hidden_state'])
bert_emb_xtest = np.array(reuters_encoded['test']['hidden_state'])

# Save as .npy
np.save("bert_embedded_test", bert_emb_xtest)

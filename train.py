import threading

import bilstmCRF.util as util
import torch
from bilstmCRF.bistmCRF import BiLSTM_CRF
from bilstmCRF.util import prepare_sequence
import torch.optim as optim
import random
import time

#device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def train(model, epoch, optimizer,training_batch, word_to_ix, tag_to_ix):
    start_time = time.time()
    model.train()  # Turn on the train mode
    total_loss = 0.

    for i, pair in enumerate(training_batch):
        sentence, tags = eval(pair[0]), eval(pair[1])
        if not sentence : continue

        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()
        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)


        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)
        # Step 4. Compute the loss, gradients, and update the parameters by
        # # calling optimizer.step()
        # loss.backward()
        # optimizer.step()
        total_loss += loss.item() / len(sentence_in)

        log_interval = 20
        if i % log_interval == 0 and i != 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | '
                  'lr {:02.6f} | ms/batch {:5.2f} | '
                  'loss {:5.5f} |'.format(epoch, lr,
                elapsed * 1000 / log_interval,
                cur_loss))
            total_loss = 0
            start_time = time.time()

        # calling optimizer.step()
        loss.backward()
        optimizer.step()



def evaluate(eval_model, data_source, word_to_ix, tag_to_ix):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for x_k, y_k in data_source:
            #print(x_k)
            x_k = eval(x_k)
            y_k = eval(y_k)
            if not x_k: continue
            sentence_k = prepare_sequence(x_k, word_to_ix)
            targets_k = torch.tensor([tag_to_ix[t] for t in y_k], dtype=torch.long)
            loss_test = eval_model.neg_log_likelihood(sentence_k, targets_k)
            total_loss += float(loss_test) / len(sentence_k)
    return total_loss



TRAIN_DATA_PATH = util.TRAIN_DATA_PATH
START_TAG = util.START_TAG
STOP_TAG = util.STOP_TAG
UNK_TAG = util.UNK_TAG

EPOCHES = util.EPOCHES
EMBEDDING_DIM = util.EMBEDDING_DIM
HIDDEN_DIM = util.HIDDEN_DIM
BATCH_SIZE = util.BATCH_SIZE
lr = util.lr

MODEL_PATH = util.MODEL_PATH


def trainning(TRAIN_DATA_PATH):
    training_data, word_to_ix, tag_to_ix = util.data_prepare(TRAIN_DATA_PATH)
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_model = None
    test_batch_data = [random.choice(training_data[-5000:]) for _ in range(1000)]

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(EPOCHES):  # again, normally you would NOT do 300 epochs, it is toy data
        training_batch = [random.choice(training_data[:-5000]) for _ in range(BATCH_SIZE)]
        epoch_start_time = time.time()
        train(model,epoch,optimizer,training_batch,word_to_ix,tag_to_ix)
        val_loss = evaluate(model, test_batch_data,word_to_ix,tag_to_ix)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '.format(epoch, (time.time() - epoch_start_time),
                                         val_loss))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        if epoch % 20 == 0:
            torch.save(best_model.state_dict(), MODEL_PATH)



    #存储数据


import sys
if __name__ == '__main__':
    sys.setrecursionlimit(100000)
    threading.stack_size(200000000)
    thread = threading.Thread(target=trainning(TRAIN_DATA_PATH))
    thread.start()






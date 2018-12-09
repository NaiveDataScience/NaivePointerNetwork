import numpy as np

##################
# ???
from model import PointerNetwork
from model import to_var
# ???
####################
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

MAX_EPOCH = 1000


#############################
#?????   construct your model

# model = Pointer_network(     )

#?????
#############################


batch_size = 256
# optimizer = torch.optim.Adam(params= model.parameters() ,lr= 0.001)
# loss_fun = torch.nn.CrossEntropyLoss()


mapping = {
    1 : 5,
    2 : 10,
    3 : 5
}



def getdata(shiyan=1, batch_size=256):
    if shiyan == 1:
        high = 100
        senlen = 5
        x = np.array([np.random.choice(range(high), senlen, replace=False) for _ in range(batch_size)])
        y = np.argsort(x)
    elif shiyan == 2:
        high = 100
        senlen = 10
        x = np.array([np.random.choice(range(high), senlen, replace=False) for _ in range(batch_size)])
        y = np.argsort(x)
    elif shiyan == 3 :
        senlen = 5
        x = np.array([np.random.random(senlen) for _ in range(batch_size)])
        y = np.argsort(x)
    return x,y

def evaluate(model, index, epoch=300):
    accuracy_sum = 0.0
    for i in range(epoch):

        test_x ,test_y = getdata(shiyan = index)
        test_x = torch.LongTensor(test_x)
        test_y = torch.LongTensor(test_y)

        ###############################
        # ????
        predict_y = model(test_x)

        _v, indices = torch.max(predict_y, 2)
        correct_count = sum([1 if torch.equal(ind.data, y.data) else 0 for ind, y in zip(indices, test_y)])
        # compute prediction ,and then get the accuracy
        #############################

        accuracy = correct_count / len(test_x)
        accuracy_sum += accuracy
    print('accuracy is ',accuracy_sum / epoch)


def evaluate_naive(model, test_x, test_y):

    predict_y = model(test_x)

    _v, indices = torch.max(predict_y, 2)
    correct_count = sum([1 if torch.equal(ind.data, y.data) else 0 for ind, y in zip(indices, test_y)])
    # compute prediction ,and then get the accuracy
    #############################   
    accuracy = correct_count / len(test_x)
    print("accuracy is " + str(accuracy))

def pipeline(shiyan, weight_size=256, dataset_size=2500, batch_size=100):
    input_seq_len = mapping[shiyan]
    model = PointerNetwork(100, 1, input_seq_len)
    # model.train()
    optimizer = optim.Adam(model.parameters())
    train_X, train_Y = getdata(shiyan, dataset_size)
    train_x = torch.LongTensor(train_X)
    train_y = torch.LongTensor(train_Y)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size)
    for epoch in range(MAX_EPOCH):
        for x_batch, y_batch in loader:
        ############################
        # compute the  prediction
            probs = model(x_batch)
            outputs = probs.view(-1, input_seq_len)
            y_batch = y_batch.view(-1)
            loss = F.nll_loss(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.save(model, 'mytraining' + str(shiyan) + '.pkl')

        if epoch % 10 == 0 and epoch != 0:
            print (epoch ,' \t loss is \t',loss.item())
            print (loss)



        if epoch % 100 == 0 and epoch != 0: #
            evaluate(model, shiyan)
            evaluate_naive(model, train_x, train_y)

if __name__ == '__main__':
    for index in range(1, 4):
        pipeline(index)


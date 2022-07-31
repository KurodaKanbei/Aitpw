import fasttext
import os

def eval(model):
    val = 'ft_val.txt'
    cnt = 0
    tot = 0
    with open(os.path.join('data', val)) as f:
        for line in f.readlines():
            line = line.strip()
            tot += 1
            # print(str(cnt) + ' ' + line)
            label = model.predict(line.split(' ', 1)[1])[0][0]
            # print(label)
            if label == line.split(' ', 1)[0]:
                cnt += 1
    print('val acc = %f' % (cnt / tot))
    return cnt / tot

def pred(model):
    test = 'ft_test.txt'
    dest = 'ft_pred.csv'
    f_pred = open(os.path.join('predictions', dest), 'w')
    f_pred.write('Id,Prediction\n')
    tot = 0
    with open(os.path.join('data', test), 'r') as f:
        for line in f.readlines():
            line = line.strip()
            tot += 1
            label = 2 * int(model.predict(line)[0][0][-1]) - 1
            f_pred.write('%d,%d\n' % (tot, label))
    f_pred.close()

if __name__ == '__main__':
    source = 'ft_train.txt'
    acc_list = []
    for idx in range(5):
        model = fasttext.train_supervised(input=os.path.join('data', source), epoch=10)    
        model.save_model(os.path.join('models', 'ft_model.bin'))
        acc = eval(model)
        if len(acc_list) == 0 or acc > max(acc_list):
            pred(model)
        acc_list.append(acc)
    acc_list = sorted(acc_list)
    print(sum(acc_list) / len(acc_list)) 
    print(acc_list)

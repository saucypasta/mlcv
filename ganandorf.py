import numpy as np
import tensorflow as tf
from PIL import Image
import gan
import random as ra
import argparse



def run_models(sims, d_model, t_model, faked_count):
    f = 0
    while len(t_model.memory) < faked_count:
        if f % 50 == 0:
            print("Length of trickster memory: ",len(t_model.memory))
        t_model.train_trickster(d_model)
    print("finished creating trickster memory")
    for i in range(0, sims):
        mem_inds = np.random.choice(np.shape(t_model.memory)[0],faked_count)
        print("finisehd finding random noise index")
        memx_data = [t_model.memory[j] for j in mem_inds]
        memy_data = np.ones(faked_count) * 10
        print("finisehd creating random data")
        while (d_model.train_on_batch(memx_data, memy_data,batch_size=500) < .9):
            pass

        print("finished trainig classifier")

        j = 0
        while(t_model.train_trickster(d_model, batch_size =1 ,flag=j%3==0) > .01):
           j+=1
           # t_model.train_trickster(d_model, flag=i%10==0)
        print("finished training trickster")
        #t_model.validate(d_model)
        #print("trick loss: ", t_model.loss)
        #print("Most accurates: ", np.amax(t_model.out,1))

        print("Saving model")
        dig_model.save_model()
        t_model.save_model()
        print("Finished saving model")
        show_img(t_model,dig_model)
        print("Finished showing image")
        #show_img(t_model)


def show_img(t_model,dig_model):
    input = []
    num_nums = 1
    for i in range(num_nums):
        input.append([i])

    out = t_model.model.predict(np.random.uniform(size=(num_nums,t_model.input_shape)))
    probs = dig_model.model(out)
    print("probabilites: ",probs)
    imgs = []
    for i in range(0,28):
        row = []
        for img in out:
            row.extend(img[i])
        imgs.append(row)

    print(np.shape(imgs))
    tmp = np.reshape(out, (28,28*num_nums))
    img = Image.fromarray(np.uint8(np.array(imgs) * 255) , 'L')
    img.show()


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trick_name",dest="trick_name",type=str,default="multi_faked")
    parser.add_argument("--dig_name",dest="dig_name",type=str,default="mult_classifier")

    parser.add_argument('--load_dig',dest='load_dig',action='store_true',default = False)
    parser.add_argument('--load_trick',dest='load_trick',action='store_true',default = False)
    parser.add_argument('--show_image',dest='show_image',action='store_true',default = False)
    parser.add_argument('--test_trick',dest='test_trick',action='store_true',default = False)




    return parser.parse_args()

args = parse_argument()

trick_model = gan.trickster(name=args.trick_name,input_shape=1, digit_type=9)
dig_model = gan.digit_model(name=args.dig_name)
if(args.load_dig):
    dig_model.load_model()
if args.load_trick:
    trick_model.load_model()
if args.test_trick:
    trick_model.train_trickster(dig_model, batch_size =3,flag=True)
    exit(0)
if args.show_image:
    show_img(trick_model, dig_model)
    exit(0)
run_models(1000, dig_model, trick_model, 500)
trick_model.save_model()
dig_model.save_model()

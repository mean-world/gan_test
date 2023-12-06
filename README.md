# gan_test
Implemented by reference papers Stock Market Prediction Based on Generative Adversarial Network
https://www.sciencedirect.com/science/article/pii/S1877050919302789

Based on this, we try to use CNN in the D network part.

Generator: LSTM model
Discriminator: MLP or CNN model
```
#in paper_code_test file 
python train.py -e[epochs] -m[D net model(mlp or cnn)] -t[timewindow] -d[dataset] -l1[hyper-parameters λ1] -l2[hyper-parameters λ1] -b[batch size] -lr[learning rate] -be[beta1] -s[split dataset rate]
```

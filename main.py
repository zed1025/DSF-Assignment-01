'''
Amit Kumar
Roll No: 22CSM1R02
MTech-CSE, Sem01
DSF Assignment 1
'''

from amit import utils
from amit import logistic, knn, decision_tree, lda, naive_bayes


lr = logistic.LR()
# print(lr[0])
# lr_final = utils.find_outputs(lr)
lr_final = [utils.find_outputs(lr[i]) for i in range(len(lr))]
# print(lr_final)
# print('\n\n')

nb = naive_bayes.NB()
# print(nb[0])
nb_final = [utils.find_outputs(nb[i]) for i in range(len(nb))]
# print(nb_final)
# print('\n\n')

lda = lda.LDA()
# print(lda[0])
lda_final = [utils.find_outputs(lda[i]) for i in range(len(lda))]
# print(lda_final)
# print('\n\n')

knn = knn.KNN()
# print(knn[0])
knn_final = [utils.find_outputs(knn[i]) for i in range(len(knn))]
# print(knn_final)
# print('\n\n')

dt = decision_tree.DT()
# print(dt[0])
dt_final = [utils.find_outputs(dt[i]) for i in range(len(dt))]
# print(dt_final)
# print('\n\n')

ans = utils.printOutputToFile(lr_final, nb_final, lda_final, knn_final, dt_final)
for i in ans:
    print(i)
    print('\n\n')

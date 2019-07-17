import numpy as np
M = np.random.randint(20, 50, (55, 1))
r = M.sum(axis=1).reshape(55,1)
M = M/r
M = M.sum(axis=0)
print(np.vstack((M,M)))
percents = np.array((0.6,0.2,0.2))

if percents.shape[0]==3:#Treino, validação e teste
	percents_train = [percents[0]]*5
	percents_test = [percents[1]]     
	percents_val = [percents[2]] 
	percents_all = np.hstack((percents_train, percents_test, percents_val))

print(M.T - percents_all)
print()


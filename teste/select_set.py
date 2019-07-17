
import numpy as np
import cvxpy as cp
import cvxopt as co

# M poços e 5 tipos de falhas

# M = [
# 	poço_0 := (arraste_0, kick_0, perda_0, prisão_0, topada_0)
# 	poço_1 := (arraste_1, kick_1, perda_1, prisão_1, topada_1)
# 	poço_2 := (arraste_2, kick_2, perda_2, prisão_2, topada_2)
# 	... ... ...
# ]

def partitions(M: np.ndarray, percents=None, verbose=0):
	N = M.shape[0] #matriz com valores das falhas em cada poço

	##Dizendo a quantidade de dados totais em cada conjunto (treino, validação e teste) por falha
	if percents is None:
		percents = np.array((0.6,0.2,0.2))

	if percents.shape[0]==3:#Treino, validação e teste
		percents_train = percents[0] * M.sum(axis=0)
		percents_test = percents[1] * M.sum(axis=0)
		percents_val = percents[2] * M.sum(axis=0)
		percents_all = np.vstack((percents_train, percents_test, percents_val))
	else:#Treino e teste
		percents_train = percents[0] * M.sum(axis=0)
		percents_test = percents[1] * M.sum(axis=0)
		percents_all = np.vstack((percents_train, percents_test))

	##Variaveis do problema
	ss = cp.Variable((percents.shape[0], 5), name="set_total")  #Matriz com a quantidade total de falhas, onde cada entrada se refere: linha->conjunto e coluna->falha

	z0 = cp.Variable(N, name="is_from_0", boolean=True) #vetor binário, onde z0_i=1 significa que o poço i está no treino 
	z1 = cp.Variable(N, name="is_from_1", boolean=True) #vetor binário, onde z1_i=1 significa que o poço i está no teste
	if percents.shape[0]==3:
		z2 = cp.Variable(N, name="is_from_2", boolean=True) #vetor binário, onde z1_i=1 significa que o poço i está no validação

	##Restrições
	if percents.shape[0]==3:#Cada poço deve está somente em um conjunto
		constraints = [
		z0 + z1 + z2 == 1,
		
		#porção exigida nos conjuntos
		# sum(z0.T)/N == percents[0],
		# sum(z1.T)/N == percents[1], 
		# sum(z2.T)/N == percents[2], 

		ss[0, 0] == M[:, 0] @ z0,  #nº arraste no treino
		ss[1, 0] == M[:, 0] @ z1,  #nº arraste no teste
		ss[2, 0] == M[:, 0] @ z2,  #nº arraste no validação
	
		ss[0, 1] == M[:, 1] @ z0,  #nº kick no treino
		ss[1, 1] == M[:, 1] @ z1,  #nº kick no teste
		ss[2, 1] == M[:, 1] @ z2,  #nº kick na validação

		ss[0, 2] == M[:, 2] @ z0,  #nº perda no treino
		ss[1, 2] == M[:, 2] @ z1,  #nº perda no teste
		ss[2, 2] == M[:, 2] @ z2,  #nº perda na validação 

		ss[0, 3] == M[:, 3] @ z0,  #nº prisão no treino
		ss[1, 3] == M[:, 3] @ z1,  #nº prisão no teste
		ss[2, 3] == M[:, 3] @ z2,  #nº prisão na validação

		ss[0, 4] == M[:, 4] @ z0,  #nº topada no treino
		ss[1, 4] == M[:, 4] @ z1,  #nº topada no teste
		ss[2, 4] == M[:, 4] @ z2,  #nº topada na validação
	]
	else:
		constraints = [
		z0 + z1 == 1,
		
		#porção exigida nos conjuntos
		sum(z0.T)/N == percents[0], 
		sum(z1.T)/N == percents[1], 

		ss[0, 0] == M[:, 0] @ z0,  #nº arraste no treino
		ss[1, 0] == M[:, 0] @ z1,  #nº arraste no teste
	
		ss[0, 1] == M[:, 1] @ z0,  #nº kick no treino
		ss[1, 1] == M[:, 1] @ z1,  #nº kick no teste

		ss[0, 2] == M[:, 2] @ z0,  #nº perda no treino
		ss[1, 2] == M[:, 2] @ z1,  #nº perda no teste

		ss[0, 3] == M[:, 3] @ z0,  #nº prisão no treino
		ss[1, 3] == M[:, 3] @ z1,  #nº prisão no teste

		ss[0, 4] == M[:, 4] @ z0,  #nº topada no treino
		ss[1, 4] == M[:, 4] @ z1,  #nº topada no teste
	]

	##resolvendo o problema
	objective = cp.norm1(ss - percents_all)
	p = cp.Problem(cp.Minimize(objective), constraints)
	#p.solve(solver=cp.GLPK_MI)  #Outro método
	p.solve(solver=cp.ECOS_BB, mi_max_iters=5000, mi_abs_eps=1e-6, mi_rel_eps=1e-3, verbose=(verbose >= 2))
	if verbose: print("PROBLEM STATUS:", p.status)

	#erro obtido para cada falha
	err_0 = (ss.value[:,0])/M[:,0].sum(axis=0)
	err_1 = (ss.value[:,1])/M[:,1].sum(axis=0)
	err_2 = (ss.value[:,2])/M[:,2].sum(axis=0)
	err_3 = (ss.value[:,3])/M[:,3].sum(axis=0)
	err_4 = (ss.value[:,4])/M[:,4].sum(axis=0)
	err   = np.vstack((err_0, err_1, err_2, err_3,  err_4))

	z0 = np.asarray(z0.value+.1, dtype=int)
	z1 = np.asarray(z1.value+.1, dtype=int)
	if percents.shape[0]==3:
		z2 = np.asarray(z2.value+.1, dtype=int)
		
	##retorno
	if percents.shape[0]==3:
		if verbose: print(np.max(err-percents), ";  errors:", [err_0, err_1, err_2, err_3,  err_4], \
			"; quantidade de pocos em cada conjunto:", [np.sum(z0),np.sum(z1),np.sum(z2)])
		return (ss.value[:,0], ss.value[:,1], ss.value[:,2], ss.value[:,3], ss.value[:,4], z0, z1, z2)
	else:
		if verbose: print(np.max(err-percents), ";  errors:", [err_0, err_1, err_2, err_3,  err_4], \
			"; quantidade de pocos em cada conjunto:", [np.sum(z0),np.sum(z1)])
		return (ss.value[:,0], ss.value[:,1], ss.value[:,2], ss.value[:,3], ss.value[:,4], z0, z1)


if __name__ == "__main__":
	np.random.seed(12345)
	M = np.random.randint(20, 50, (55, 5))
	percents = np.array([0.6, 0.2, 0.2])
	s0, s1, s2, s3, s4, z0, z1, z2 = partitions(M, percents, verbose=1)

#	print(M, end="\n\n")
#	print("\ns0: {}".format(s0), "s1: {}".format(s1), "s2: {}".format(s2), "s3: {}".format(s3), "s4: {}".format(s4), sep="\n\n", end="\n\n")
#	print("\nz0: {}".format(z0), "z1: {}".format(z1), "z2: {}".format(z2), sep="\n\n")



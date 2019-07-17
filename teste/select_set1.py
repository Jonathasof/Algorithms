import numpy as np
import cvxpy as cp
import cvxopt as co


def partitions(M: np.ndarray, percents=np.array((0.6,0.2,0.2)), verbose=0):
	""" Divide os poços em conjuntos de treinamento, teste e validação (se necessário).
		
	Args:
		M (array_like(int, ndim=2)): Matriz representando em cada linha um poço e nas colunas as falhas.
		Ex.:
		M = [
		 	poço_0 := (arraste_0, kick_0, perda_0, prisão_0, topada_0)
		 	poço_1 := (arraste_1, kick_1, perda_1, prisão_1, topada_1)
		 	poço_2 := (arraste_2, kick_2, perda_2, prisão_2, topada_2)
		 	... ... ...
		]

		percents (array_like(float, ndim=1), optional): Fração de valores para cada conjunto (treinamento, teste e validação). O padrão é retornar [0.6, 0.2, 0.2].

		verbose (int, optional): Informação adicional sobre o processo de otimização. O padrão é retornar 0 (nenhuma informação).
	
	Returns:
		S array_like(float, ndim=2): Retorna uma matriz contendo os valores em fração para cada falha (coluna) e conjunto (linha) depois do processo de otimização.

		z0 array_like(bool, ndim=1): Retorna um vetor com valores booleanos, caso o i-ésimo poço esteja no conjunto de treinamento 
		então a i-ésima entrada é igual a 1, caso contrário a entrada é igual à 0.

		z1 array_like(bool, ndim=1): Retorna um vetor com valores booleanos, caso o i-ésimo poço esteja no conjunto de teste
		então a i-ésima entrada é igual a 1, caso contrário a entrada é igual à 0.

		z2 array_like(bool, ndim=1): Retorna um vetor com valores booleanos, caso o i-ésimo poço esteja no conjunto de validação 
		então a i-ésima entrada é igual a 1, caso contrário a entrada é igual à 0.
	"""
	N = M.shape[0] #matriz com valores das falhas em cada poço 

	##Dizendo a porcentagem em cada conjunto (treino, validação e teste) 
	if percents.shape[0]==3:#Treino, validação e teste
		percents_train = [percents[0]]*5
		percents_test = [percents[1]]*5  
		percents_val = [percents[2]]*5 
		percents_all = np.vstack((percents_train, percents_test, percents_val))
	else:#Treino e teste
		percents_train = [percents[0]]*5
		percents_test = [percents[1]]*5
		percents_all = np.vstack((percents_train, percents_test))

	##Variáveis do problema
	S = cp.Variable((percents.shape[0], 5), name="set_total")  #Matriz com a porcentagem de falhas, onde cada entrada se refere: linha->conjunto e coluna->falha

	z0 = cp.Variable(N, name="is_from_0", boolean=True) #vetor booleano, onde z0_i=1 significa que o poço i está no treino 
	z1 = cp.Variable(N, name="is_from_1", boolean=True) #vetor booleano, onde z1_i=1 significa que o poço i está no teste
	if percents.shape[0]==3:
		z2 = cp.Variable(N, name="is_from_2", boolean=True) #vetor booleano, onde z1_i=1 significa que o poço i está no validação

	##Restrições
	if percents.shape[0]==3:#Cada poço deve está somente em um conjunto
		constraints = [
		z0 + z1 + z2 == 1,
		
		#porção exigida nos conjuntos (opcional)
		# sum(z0.T)/N == percents[0],
		# sum(z1.T)/N == percents[1], 
		# sum(z2.T)/N == percents[2], 

		S[0, 0] == M[:, 0]/M[:,0].sum(axis=0) @ z0,  #fração de arraste no treino
		S[1, 0] == M[:, 0]/M[:,0].sum(axis=0) @ z1,  #fração de arraste no teste
		S[2, 0] == M[:, 0]/M[:,0].sum(axis=0) @ z2,  #fração de arraste no validação
	
		S[0, 1] == M[:, 1]/M[:,1].sum(axis=0) @ z0,  #fração de kick no treino
		S[1, 1] == M[:, 1]/M[:,1].sum(axis=0) @ z1,  #fração de kick no teste
		S[2, 1] == M[:, 1]/M[:,1].sum(axis=0) @ z2,  #fração de kick na validação

		S[0, 2] == M[:, 2]/M[:,2].sum(axis=0) @ z0,  #fração de perda no treino
		S[1, 2] == M[:, 2]/M[:,2].sum(axis=0) @ z1,  #fração de perda no teste
		S[2, 2] == M[:, 2]/M[:,2].sum(axis=0) @ z2,  #fração de perda na validação 

		S[0, 3] == M[:, 3]/M[:,3].sum(axis=0) @ z0,  #fração de prisão no treino
		S[1, 3] == M[:, 3]/M[:,3].sum(axis=0) @ z1,  #fração de prisão no teste
		S[2, 3] == M[:, 3]/M[:,3].sum(axis=0) @ z2,  #fração de prisão na validação

		S[0, 4] == M[:, 4]/M[:,4].sum(axis=0) @ z0,  #fração de topada no treino
		S[1, 4] == M[:, 4]/M[:,4].sum(axis=0) @ z1,  #fração de topada no teste
		S[2, 4] == M[:, 4]/M[:,4].sum(axis=0) @ z2,  #fração de topada na validação
	]
	else:
		constraints = [
		z0 + z1 == 1,
		
		#porção exigida nos conjuntos (opcional)
		# sum(z0.T)/N == percents[0], 
		# sum(z1.T)/N == percents[1], 

		S[0, 0] == M[:, 0]/M[:,0].sum(axis=0) @ z0,  #fração de arraste no treino
		S[1, 0] == M[:, 0]/M[:,0].sum(axis=0) @ z1,  #fração de arraste no teste
	
		S[0, 1] == M[:, 1]/M[:,1].sum(axis=0) @ z0,  #fração de kick no treino
		S[1, 1] == M[:, 1]/M[:,1].sum(axis=0) @ z1,  #fração de kick no teste

		S[0, 2] == M[:, 2]/M[:,2].sum(axis=0) @ z0,  #fração de perda no treino
		S[1, 2] == M[:, 2]/M[:,2].sum(axis=0) @ z1,  #fração de perda no teste

		S[0, 3] == M[:, 3]/M[:,3].sum(axis=0) @ z0,  #fração de prisão no treino
		S[1, 3] == M[:, 3]/M[:,3].sum(axis=0) @ z1,  #fração de prisão no teste

		S[0, 4] == M[:, 4]/M[:,4].sum(axis=0) @ z0,  #fração de topada no treino
		S[1, 4] == M[:, 4]/M[:,4].sum(axis=0) @ z1,  #fração de topada no teste
	]


	##resolvendo o problema
	objective = cp.norm1(S - percents_all)
	p = cp.Problem(cp.Minimize(objective), constraints)
	#p.solve(solver=cp.GLPK_MI)  #Outro método
	p.solve(solver=cp.ECOS_BB, mi_max_iters=5000, mi_abs_eps=1e-6, mi_rel_eps=1e-3, verbose=(verbose >= 2))
	if verbose: print("PROBLEM STATUS:", p.status)

	#erro obtido para cada falha
	err_0 = (S.value[:,0]-percents)
	err_1 = (S.value[:,1]-percents)
	err_2 = (S.value[:,2]-percents)
	err_3 = (S.value[:,3]-percents)
	err_4 = (S.value[:,4]-percents)
	err   = np.vstack((err_0, err_1, err_2, err_3,  err_4))

	z0 = np.asarray(z0.value+.1, dtype=int)
	z1 = np.asarray(z1.value+.1, dtype=int)
	if percents.shape[0]==3:
		z2 = np.asarray(z2.value+.1, dtype=int)
		
	##retorno
	if percents.shape[0]==3:
		if verbose: print(p.value)
		return (S, z0, z1, z2, err)
	else:
		if verbose: print(p.value)
		return (S, z0, z1, err)


if __name__ == "__main__":
	np.random.seed(12345)
	M = np.random.randint(20, 500, (55, 5))
	percents = np.array([0.6, 0.2, 0.2])
	S, z0, z1, z2, err = partitions(M, percents, verbose=1)

	print("The maximum error:", np.max(err))
	print("\ns0: {}".format(S.value[:,0]), "s1: {}".format(S.value[:,1]), "s2: {}".format(S.value[:,2]), "s3: {}".format(S.value[:,3]), "s4: {}".format(S.value[:,4]), sep="\n\n", end="\n\n")
	print( "Quantidade de pocos em cada conjunto:", [np.sum(z0),np.sum(z1), np.sum(z2)])





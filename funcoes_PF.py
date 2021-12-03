#imports
import numpy as np
import math

def conectividade(Inc,nm,nn):
    '''
    Recebe excel incidencia e retorna matriz de conectividade 
    normal e transposta
    '''
    no_origem = Inc[:,0]
    no_destino = Inc[:,1]
    C = np.zeros((nm,nn))
    for coluna in range(len(no_origem)): 
        C[int(no_origem[coluna] -1),coluna] = -1 
        C[int(no_destino[coluna] -1),coluna] = 1
    return (C)

def membros(N,C):
    '''
    membros = N*C
    '''
    return N@C

def calcula_l(M):
    '''
    Calcula o comprimento de cada elemento a partir do
    modulo de N*C
    '''
    l = np.zeros((np.shape(M)[1], 1))
    for coluna in range(np.shape(M)[1]):
        l[coluna] = np.linalg.norm(M[:,coluna])
    return l

def calcula_trig(N,M,l):
    '''
    Recebe a matriz dos nos e dos membros e retorna o angulo entre os elementos
    '''
    t = np.zeros((np.shape(M)[1], 4))
    for i in range((np.shape(M)[1])):
        #calcula cos
        t[i,2] = (M[0,i])/l[i]
        t[i,0] = -t[i,2]
        #calcula sen
        t[i,3] = (M[1,i])/l[i]
        t[i,1] = -t[i,3]
    return t

def calcula_Ke(C, M, E, A,nn):
    '''
    Funcao que calcula cada K de cada elemento e K global
    '''
    l = calcula_l(M)
    Se = np.zeros((2, 2))
    Kg = np.zeros((nn*2,nn*2))
    lista_ke = []
    linhas = np.shape(M)[0]
    colunas = np.shape(M)[1] #igual a n de elementos
    for coluna in range(colunas):
        me = (M[:,coluna]).reshape(linhas,1)
        me_t = np.transpose(me)
        Se = ((E[coluna]*A[coluna])/l[coluna])*((me@me_t)/(np.linalg.norm(me))**2)
        ce = C[:,coluna].reshape(np.shape(C)[0],1)
        ce_t = np.transpose(ce)
        Ke = np.kron((ce@ce_t),Se)
        lista_ke.append(Ke) 
        Kg += Ke
    return Kg, Ke

def aplica_cc(Kg,R):
    if np.shape(Kg)[1] == 1:
        return np.delete(Kg, list(R[:,0].astype(int)),axis=0)
    '''
    elimino os graus de liberdade que estao com restricao 
    ao aplicar as condicoes de contorno
    '''
    return np.delete(np.delete(Kg, list(R[:,0].astype(int)),axis=0), list(R[:,0].astype(int)),axis=1)

def calcula_jacobi(a,b,tol):
    '''
    Recebe 2 vetores [A]{x}=[B] 
    Repetira ate atingir a convergencia desejada.
    '''
    linhas = np.shape(a)[0]
    colunas = np.shape(a)[1]
    x = np.zeros((linhas,1))
    xnew = np.zeros((linhas,1))
    max_i = 100
    for i in range(max_i):
        for linha in range(linhas):
            for coluna in range(colunas):
                if linha!=coluna:
                    xnew[linha] +=a[linha][coluna]*x[coluna]
            xnew[linha] = (b[linha]-xnew[linha])/a[linha][linha] 
        #print(f'x={xnew}')
        #print()

        # Erro
        err = max(abs((xnew-x)/xnew))

        # Atualizar
        x = np.copy(xnew)
        xnew.fill(0)
        if err<=tol:
            print(f'CONVERGIU EM i={i} com err = {err}')
            break
    return x

def gauss_seidel(vec1, vector_f, limit_iter=100, tolerance=1e-6):
  
    x = np.zeros((len(vec1), 1))
    D = np.diag(vec1)
    R = vec1 - np.diagflat(D)

    x_copy = x.copy()

    for n_iter in range(limit_iter):
        for j in range(vec1.shape[0]):
            x[j, 0] = np.divide((vector_f[j, 0] - np.sum(R[j, :]@x)), D[j])

        if (n_iter > 0):
            diff = vector_f - vec1@x
            dif_abs = np.abs(np.divide(diff, vector_f, where=vector_f!=0))
            rel_diff = np.max(dif_abs)

            if rel_diff <= tolerance:
                print(f"Num iterações: {n_iter} | Erro: {rel_diff}")
                return x

        x_copy = x.copy()
    print(f"Maximo de iterações: {n_iter} alcançado, com erro: {rel_diff}")
    return x

def calcula_u_comp(u,R):
    u_comp = np.zeros((len(R)+len(u),1))
    k = 0
    r = list(R[:,0].astype(int))
    for i in range(len(u_comp)):
        if i not in r:
            u_comp[i] = u[k]
            k+=1
    return u_comp

def calcula_deformacao(Inc,u_comp,l,R,trig):
    lista_d = np.zeros((len(l), 1))
    
    for i in range(len(l)):
        inicio = 2*i
        stop = 0
        
        idxs = [(int(Inc[i, 0])-1)*2, (int(Inc[i, 0])-1)*2 +1, 
                (int(Inc[i, 1])-1)*2, int(Inc[i, 1]-1)*2 +1]

        ue = u_comp[idxs]   
        d = (1/l[i])*(trig[i]@ue)
        lista_d[i] = d[0]
        
    return lista_d

def calcula_tensao(E,d):
    return [a*b for a,b in zip(E,d)]

def calcula_r_apoio(Kg, u_comp,R):
    R = list(R[:,0].astype(int))
    return ((Kg@u_comp)[R])
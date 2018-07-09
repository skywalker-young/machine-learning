import numpy as np

pi=[0.3,0.2,0.5]#3 initial states
#B=np.zeros((3,3))
B=np.matrix('0.5,0.5;0.75,0.25;0.25,0.75')
#print(B)
A=[[0.5,0.3,0.2]for i in range(3)]
A=np.matrix(A)
#print(A)
obs=[1, 1, 1, 1, 2, 1, 2, 2, 2, 2]


def forward_scale(obs):
    T=len(obs)
    N=3#5 states
    alpha=np.zeros([N,T]) #N rows T column
    scale = np.zeros(T)
    #return alpha,scale
    alpha[:,0]=pi[:]*B[:,obs[0]-1] ###why obs[0]-1
    scale[0]=np.sum(alpha[:,0])
    #return scale
    alpha[:,0]=alpha[:,0]/scale[0]

    for t in range(1,T):
        for n in range(0,N):
            alpha[n,t]=np.sum(alpha[:,t-1]*A[:,n])*B[n,obs[t]-1] #why obs[t]-1
        scale[t]=np.sum(alpha[:,t])
        alpha[:,t]=alpha[:,t]/scale[t]
    logprob=np.sum(np.log(scale[:]))
    return logprob,alpha,scale


#print(forward_scale(obs))
#print(obs[0])
#####################################
####################################
def backward_scale(obs,scale):
  T=len(obs)
  N=3 #3 states
  beta=np.zeros([N,T])
  beta[:,T-1]=1/scale[T-1]
  for t in reversed(range(0,T-1)):
      for n in range(0,N):
          beta[n,t]=np.sum(B[:,obs[t+1]-1]*A[n,:]*beta[:,t+1])
          beta[n,t]/=scale[t]
  return beta

def compute_gamma(alpha,beta):
    gamma=np.zeros(alpha.shape)
    gamma=alpha[:,:]*beta[:,:]
    gamma=gamma/np.sum(gamma,0)
    return gamma
def compute_xi(obs,alpha,beta):
    T=len(obs)
    N=3
    xi=np.zeros((N,N,T-1))#N rows T-1 columns multiple N
    for t in range(0,T-1):
        for i in range(0,N):
            for j in range(0,N):
                xi[i,j,t]=alpha[i,t]*A[i,j]*B[j,obs[t+1]-1]*beta[j,t+1]
        xi[:,:,t]/=np.sum(np.sum(xi[:,:,t],1),0)
    return xi

def baum_welch(obs):
    T=len(obs)
    M=5
    N=3
    alpha=np.zeros([N,T])
    beta=np.zeros([N,T])
    scale=np.zeros(T)
    gamma=np.zeros([N,T])
    xi=np.zeros([N,N,T-1])
    logprobprev,alpha,scale=forward_scale(obs)
    beta=backward_scale(obs,scale)
    gamma=compute_gamma(alpha,beta)
    xi=compute_xi(obs,alpha,beta)
    logprobinit=logprobprev

    while True:
        pi=0.001+0.999*gamma[:,0]
        for i in range(N):
            denominator=np.sum(gamma[i,0:T-1])
            for j in range(N):
                numerator=np.sum(xi[i,j,0:T-1])
                A[i,j]=numerator/denominator
        A=0.001+0.999*A
        for j in range(0,N):
            denominator=np.sum(gamma[j,:])
            for k in range(0,M):
                numerator=0
                for t in range(0,T):
                    if (obs[t]==k):
                        numerator+=gamma[j,t]
                B[j,k]=numerator/denominator
        B=0.001+0.999*B

        logprobcur,alpha,scale=forward_scale(obs)
        beta=backward_scale(obs,scale)
        gamma=compute_gamma(alpha,beta)
        xi=compute_xi(obs,alpha,beta)
        delta=logprobcur-logprobprev
        if (delta<0.001):
            break
    logprobfinal=logprobcur
    return logprobinit,logprobfinal


'''
a=(np.zeros((2,2,4)))
a[0][0][0]=1
a[0][0][1]=2
a[0][1][0]=3
a[0][1][1]=99
a[1][1][3]=8
print(a)
'''
prob, alpha, scale =forward_scale(obs)
print('forward prob is %f'%prob)
beta = backward_scale(obs,scale)

logprobinit, logprobfinal = baum_welch(obs)

print('pi is %f'%pi)
print('A is %f'%A)
print('B is %f'%B)
print('initial prob is %f'%logprobinit)

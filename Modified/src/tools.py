#!/usr/bin/env python3
import numpy as np

def make_array(nCells, seqLength, CellPoints, nVars,nVarsL):
  if nCells:
    dataset = np.ndarray((nCells, seqLength, CellPoints, CellPoints, CellPoints, nVars), dtype=np.float32)
    labels =  np.ndarray((nCells, seqLength, CellPoints, CellPoints, CellPoints,nVarsL),  dtype=np.float32)
  else:
    dataset,labels = None,None
  return dataset,labels


def sortArray(array,a,start,end,position):
  array[:,:,:,:,:,position] = a[start:end,:,:,:,:]
  return array


def reshapeArray(array,CellPoints):
    vecLen = array.shape[1]
    cellLen = array.shape[0]
    if array.ndim == 5:
        varLen = array.shape[4]
        array = array.reshape(-1,CellPoints,CellPoints,CellPoints,varLen)
    else:
        array = array.reshape(-1,CellPoints,CellPoints,CellPoints,1)
    return array

def consToPrim(rho,ru,rv,rw,e):
    u = ru/rho
    v = rv/rho
    w = rw/rho
    p = 0.4*(e - 0.5*rho*(u**2+v**2+w**2))
    return u,v,w,p

def LGL_weights_3D(N):
  WGP  = LGL_weights(N)
  wIJK = np.ndarray((1,N+1,N+1,N+1,1),dtype=np.float32)
  for i in range(N+1):
    for j in range(N+1):
      for k in range(N+1):
        wIJK[0,i,j,k,0] = WGP[i]*WGP[j]*WGP[k]
    print("WGP:{}".format(WGP)) 
  print("wIJK length:{}".format(len(wIJK)))     
  print("wIJK value:{}".format(wIJK))      
  return wIJK

def LGL_weights(N):
    def qAndLEvaluation(N,x):
      L_Nm2=1.
      L_Nm1=x
      Lder_Nm2=0.
      Lder_Nm1=1.
      for iLegendre in range(2,N+1):
        L=((2*iLegendre-1)*x*L_Nm1 - (iLegendre-1)*L_Nm2)/(iLegendre)
        Lder=Lder_Nm2 + (2*iLegendre-1)*L_Nm1
        L_Nm2=L_Nm1
        L_Nm1=L
        Lder_Nm2=Lder_Nm1
        Lder_Nm1=Lder
      q=(2*N+1)/(N+1)*(x*L -L_Nm2) #L_{N+1}-L_{N-1} #L_Nm2 is L_Nm1, L_Nm1 was overwritten#
      qder= (2*N+1)*L             #Lder_{N+1}-Lder_{N-1}
      print("L value:{}".format(L))
      print("q value:{}".format(q))
      print("qder value:{}".format(qder))
      return L,q,qder
    print("N value:{}".format(N))
    
    wGP = np.ndarray(N+1,dtype=np.float32)
    xGP = np.ndarray(N+1,dtype=np.float32)
    wGP[0] = 2./(N*(N+1))
    wGP[N] = wGP[0]
    xGP[0]=-1.
    xGP[N]= 1.
    cont1=np.pi/N
    cont2=3./((8*N)*np.pi)
    nIter = 100
    Tol = 1E-15
    for iGP in range(1,int((N+1)/2-1)+1): #since points are symmetric, only left side is computed
      xGP[iGP]=-np.cos(cont1*((iGP)+0.25)-cont2/((iGP)+0.25)) #initial guess
      # Newton iteration
      for iter in range(nIter):
        L,q,qder= qAndLEvaluation(N,xGP[iGP])
        dx=-q/qder
        xGP[iGP]=xGP[iGP]+dx
        if abs(dx) <= Tol*abs(xGP[iGP]):
          break
      L,q,qder = qAndLEvaluation(N,xGP[iGP])
      xGP[N-iGP]=-xGP[iGP]
      wGP[iGP]=wGP[0]/(L*L)
      wGP[N-iGP]=wGP[iGP]
    return wGP

# shuffle dataset
def randomize(dataset, labels):
  np.random.seed()
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,...]
  shuffled_labels  = labels[permutation ,...]
  return shuffled_dataset, shuffled_labels

def reject_outliers(data, m=3):
    mask = (abs(data - np.mean(data)) < m * np.std(data)).all(axis=(1,2,3))
    return mask

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 19:57:53 2023

@author: rafai
"""


from itertools import combinations,permutations,product
import functools as ft
import numpy as np
from scipy.optimize import minimize,basinhopping,differential_evolution
import matplotlib.pyplot as plt
def constraint1(x):
    return -x[1]+2*np.pi
def constraint2(x):
    return x[1]
def constraint3(x):
    return x[0]
def constraint4(x):
    return -x[0]+np.pi/2
def constraint5(x):
    return -x[3]+2*np.pi
def constraint6(x):
    return x[3]
def constraint7(x):
    return x[2]
def constraint8(x):
    return -x[2]+np.pi/2
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
con3 = {'type': 'ineq', 'fun': constraint3}
con4 = {'type': 'ineq', 'fun': constraint4}
con5 = {'type': 'ineq', 'fun': constraint5}
con6 = {'type': 'ineq', 'fun': constraint6}
con7 = {'type': 'ineq', 'fun': constraint7}
con8 = {'type': 'ineq', 'fun': constraint8}
cons=[con1,con2,con3,con4,con5,con6,con7,con8]
def cphaseforeveryedge(N,con,tar,theta):
    seq1=[]
    seq2=[]
    seq3=[]
    seq4=[]
    sz=np.array([[1,0],[0,-1]],dtype=complex)
    for i in range(N):
        if i!=con and i!=tar:
         seq1.append(np.eye(2))
         seq2.append(np.eye(2))
         seq3.append(np.eye(2))
         seq4.append(np.eye(2))

        if i==con:
            seq1.append(np.eye(2))
            seq2.append(sz)
            seq3.append(np.eye(2))
            seq4.append(sz)
        if i==tar:
            seq1.append(np.eye(2))
            seq2.append(np.eye(2))
            seq3.append(sz)
            seq4.append(sz)
    return (ft.reduce(np.kron,seq3)+ft.reduce(np.kron,seq2))*(1/4-np.exp(1j*theta)/4)+(3/4 + np.exp(1j*theta)/4)*ft.reduce(np.kron,seq1)+(-1/4 + np.exp(1j*theta)/4)*ft.reduce(np.kron,seq4)
#list(product(axx,axx,axx)))
def initial(n):
    basisstate=[]
    for i in range(n):
      #  if array[i]==0:
            basisstate.append([[1/np.sqrt(2)],[1/np.sqrt(2)]])
       # elif array[i]==1:
       #     basisstate.append([[0],[1]])
       # else:
       #     pass
    y=(ft.reduce(np.kron,basisstate))  
    return np.array(y).flatten()

def meas(rho,v,n,tha,fa,thb,fb):
    aa=[i for i in range(2*n)]
    bb=[i for i in range(2*n)]
    for i in range(len(v)):
     aa[v[i]+n]=v[i]
     bb[v[i]+n]=v[i] 
    aa.pop(v[0])
    aa.pop(v[0]+n-1)
    aa.pop(v[1]-2)
    aa.pop(v[1]+n-3)
    allmatr=[]
    meas=np.array([np.cos(tha),np.sin(tha)*np.exp(1j*fa)])
    meas2=np.array([np.cos(thb),np.sin(thb)*np.exp(1j*fb)])
    for i in range(n):
        if i==v[0]:
            allmatr.append(np.outer(meas,meas.conj().T))
        elif i==v[1]:    
            allmatr.append(np.outer(meas2,meas2.conj().T))
          #  print(111,np.outer(meas,meas.conj().T))
        else:
            allmatr.append(np.eye(2))
          #  print(222)
    measope=ft.reduce(np.kron,allmatr)
#    print(measope)
    #print(aa,bb)
    rhotel=measope.dot(rho.dot(measope.conj().T))/np.trace(measope.dot(rho.dot(measope.conj().T)))
    rhotel=rhotel.reshape([2 for i in range(2*n)])    
 #   print(np.shape(rhotel),bb,aa)    
    return np.einsum(rhotel,bb,aa).reshape(2**(n-2),2**(n-2))
#print(meas(np.random.rand(64,64),2,6,0.2,0.2))
#print()
def paulisall(rho,n):
    sx=np.array([[0,1],[1,0]])
    sy=np.array([[0,-1j],[1j,0]])
    sz=np.array([[1,0],[0,-1]])
    ide=np.eye(2)
    axx=[ide,sx,sy,sz]
    tell=[axx,axx,axx,axx]
    y1=list(product(axx,repeat=n))
    stringlist=list(product(['i','x','y','z'],repeat=n))
    yall=[ft.reduce(np.kron,y1[i]) for i in range(4**n)]
    coeff=np.zeros(4**n)
    #print(yall[0])
    ss=[]
    nontriv=np.zeros(n)
    for i in range(4**n):
        coeff[i]=np.real(np.trace(np.dot(yall[i],rho)))/2**n
    for i in range(4**n):
        if abs(coeff[i])>10**(-15):
            ss.append([coeff[i],"".join(stringlist[i])])
            b=("".join(stringlist[i])).count('i')
            if b!=n:
             nontriv[b]+=(4**n)*(coeff[i])**2
    return coeff,ss,nontriv
#def paulidec (rho,a):
'''
from qutip import *
ghz=ket2dm(ghz_state(2))
#     pass
allcoef=[]
#a4=[]
#a3=[]
#a2=[]
#a1=[]
nn=6
fi=[]
a=[[] for i in range(nn)]
ij=0
rhoafter=[[] for i in range(20)]
bounds = [(0, np.pi), (0, 2*np.pi),(0, np.pi),(0, 2*np.pi)]
for th in np.linspace(0,np.pi,20):
 def a1(x):
   cphaseall=np.eye(2**nn,2**nn,dtype=complex)
   for i in range(nn-2):
    cphaseall*=cphaseforeveryedge(nn, i, i+1,th)  
   cphaseall*=cphaseforeveryedge(nn, 0, 4,th)
   cphaseall*=cphaseforeveryedge(nn, 2, 5,th)
   cphaseall*=cphaseforeveryedge(nn, 3, 5,th)
   cphaseall*=cphaseforeveryedge(nn, 4, 5,th)
   
   psi=cphaseall.dot(initial(nn))
   #print(psi)
   rho=np.outer(psi,psi.conj().T)
   rhotel=meas(rho,[1,5],nn,x[0],x[1],x[2],x[3])
   rhoafter[ij].append(rhotel)
   coeff,string,non=paulisall(rhotel,nn-2)
   non=list(reversed(non))
   for i in range(nn-2):
    a[i].append(non[i])
   #print(a[2],non[3]) 
   #a3.append(non[1])
   #a2.append(non[2])
   #a1.append(non[3])
   allcoef.append(coeff)
   return abs(non[0])+abs(2-non[1])+abs(8-non[2])+abs(5-non[3])#+abs(8-non[4])   
#print(allcoef)
 yall=differential_evolution(a1,bounds=bounds)#,seed=20)#x0=np.random.normal(loc=np.pi/4,scale=np.pi,size=4),constraints=cons)
             #      minimizer_kwargs = {"method":"COBYLA","constraints":cons})
#,constraints=cons)
 fi.append(yall.fun)
 print(yall,yall.x[0],yall.x[1],a[0][-1],a[1][-1],a[2][-1],a[3][-1])#,fidelity(Qobj(rhoafter[ij][-1],dims=[[2,2],[2,2]]),ghz))
 print('\n')
 #,\
 #      concurrence(Qobj(rhoafter[ij][-1],dims=[[2,2],[2,2]])))
 ij+=1
plt.plot(np.linspace(0,np.pi,20),fi) 
#print(a4)   
#forrangecoef=[]
#for j in range(4**nn):
# forrangecoef.append([allcoef[i][j] for i in range(20)])
'''



#print(a)
nn=4
a=[[] for i in range(nn)]
b=[[] for i in range(nn)]
c=[[] for i in range(nn)]
psi1=np.ones(2**nn)
psi1=psi1/np.linalg.norm(psi1)
#rho1=np.outer(psi1,psi1.conj())
#coeff,string,non=paulisall(rho1,nn)
#for i in range(nn):
#    a[i].append(list(reversed(non))[i])  
j=0  
for th in np.linspace(0,np.pi,20):
   cphaseall=np.eye(2**nn,2**nn,dtype=complex)
   cphaseall1=np.eye(2**nn,2**nn,dtype=complex)
   cphaseall2=np.eye(2**nn,2**nn,dtype=complex)
   #for i in range(nn-1):
   cphaseall*=cphaseforeveryedge(nn, 0, 1, th)  
   #cphaseall*=cphaseforeveryedge(nn, 1, 2, th)  
   cphaseall*=cphaseforeveryedge(nn, 2, 3, th)
   for i in range(3):
       cphaseall1*=cphaseforeveryedge(nn, i, i+1, th)
   for iu in range(1,4):
       cphaseall2*=cphaseforeveryedge(nn, 0, iu, th) 
       print(iu)
   psitel=cphaseall.dot(psi1)
   psitel2=cphaseall1.dot(psi1)
   psitelghz=cphaseall2.dot(psi1)
   rho1=np.outer(psitel,psitel.conj())
   rho2=np.outer(psitel2,psitel2.conj())
   rho3=np.outer(psitelghz,psitelghz.conj())
   coeff,string,non=paulisall(rho1,nn)
   coeff2,string2,non2=paulisall(rho2,nn)
   coeff3,string3,non3=paulisall(rho3,nn)
   #print(non)
   for i in range(nn):
    a[i].append(list(reversed(non))[i])
    b[i].append(list(reversed(non2))[i])
    c[i].append(list(reversed(non3))[i])
   print(a[0][j]+a[1][j]+a[2][j]+a[3][j])
   print(b[0][j]+b[1][j]+b[2][j]+b[3][j])
   print(c[0][j]+c[1][j]+c[2][j]+c[3][j])
   j+=1
g = plt.figure(1,figsize=(10, 7))
ax2 = plt.axes()
ax2.plot(np.linspace(0,1,20),a[0],'b',label=r'$A_{%d,bell}$' %(1))#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
ax2.plot(np.linspace(0,1,20),b[0],'r--',label=r'$A_{%d,wgs}$' %(1))#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
ax2.plot(np.linspace(0,1,20),c[0],'g',label=r'$A_{%d,GHZ}$' %(1))#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
#ax2.plot(np.linspace(0,1,100),a[1],label='$A_2$')#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0.01$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
#ax2.plot(np.linspace(0,1,100),a[2],label='$A_3$')#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0.02$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
#ax2.plot(np.linspace(0,1,100),a[3],label='$A_4$')#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0.05$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
#ax2.plot(range(1,nt+2),j7bc300,color=colors[4],label=r'$J_z=1\,$''MHz')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
ax2.set_xlabel(r'$\phi$',fontsize=25)
ax2.set_ylabel(r'$\rm{SLs}$',fontsize=25)#\langle (-1)^{nT}S^{satellite}_{z} \rangle\,\, density 
ax2.tick_params(labelsize=20)#ax2.set_yticks(np.arange(-0.5, 0.6, step=0.1))
#g.xticks(size=15)# bbox_to_anchor=(0.5, 1.15)
#ax2.set_title(r'$\langle (-1)^{n}\frac{1}{N-1}\sum_{k=3}^{5}S_{z,k} \rangle$')
ax2.legend(loc="best",prop={'size':20},ncol=1)#\langle (-1)^{n}\frac{1}{N-1}\sum_{k=1}^{N-1}S_{z,k} \rangle
#ax2.set_xscale('log')
ax2.grid()

h = plt.figure(2,figsize=(10, 7))
ax3 = plt.axes()
ax3.plot(np.linspace(0,1,20),a[1],'b',label=r'$A_{%d,bell}$' %(2))#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
ax3.plot(np.linspace(0,1,20),b[1],'r--',label=r'$A_{%d,wgs}$' %(2))#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
ax3.plot(np.linspace(0,1,20),c[1],'g',label=r'$A_{%d,GHZ}$' %(2))#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
#ax2.plot(np.linspace(0,1,100),a[1],label='$A_2$')#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0.01$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
#ax2.plot(np.linspace(0,1,100),a[2],label='$A_3$')#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0.02$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
#ax2.plot(np.linspace(0,1,100),a[3],label='$A_4$')#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0.05$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
#ax2.plot(range(1,nt+2),j7bc300,color=colors[4],label=r'$J_z=1\,$''MHz')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
ax3.set_xlabel(r'$\phi$',fontsize=25)
ax3.set_ylabel(r'$\rm{SLs}$',fontsize=25)#\langle (-1)^{nT}S^{satellite}_{z} \rangle\,\, density 
ax3.tick_params(labelsize=20)#ax2.set_yticks(np.arange(-0.5, 0.6, step=0.1))
#g.xticks(size=15)# bbox_to_anchor=(0.5, 1.15)
#ax2.set_title(r'$\langle (-1)^{n}\frac{1}{N-1}\sum_{k=3}^{5}S_{z,k} \rangle$')
ax3.legend(loc="best",prop={'size':20},ncol=1)#\langle (-1)^{n}\frac{1}{N-1}\sum_{k=1}^{N-1}S_{z,k} \rangle
#ax2.set_xscale('log')
ax3.grid()

hj = plt.figure(3,figsize=(10, 7))
ax4 = plt.axes()
ax4.plot(np.linspace(0,1,20),a[2],'b',label=r'$A_{%d,bell}$' %(3))#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
ax4.plot(np.linspace(0,1,20),b[2],'r--',label=r'$A_{%d,wgs}$' %(3))#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
ax4.plot(np.linspace(0,1,20),c[2],'g',label=r'$A_{%d,GHZ}$' %(3))#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
#ax2.plot(np.linspace(0,1,100),a[1],label='$A_2$')#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0.01$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
#ax2.plot(np.linspace(0,1,100),a[2],label='$A_3$')#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0.02$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
#ax2.plot(np.linspace(0,1,100),a[3],label='$A_4$')#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0.05$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
#ax2.plot(range(1,nt+2),j7bc300,color=colors[4],label=r'$J_z=1\,$''MHz')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
ax4.set_xlabel(r'$\phi$',fontsize=25)
ax4.set_ylabel(r'$\rm{SLs}$',fontsize=25)#\langle (-1)^{nT}S^{satellite}_{z} \rangle\,\, density 
ax4.tick_params(labelsize=20)#ax2.set_yticks(np.arange(-0.5, 0.6, step=0.1))
#g.xticks(size=15)# bbox_to_anchor=(0.5, 1.15)
#ax2.set_title(r'$\langle (-1)^{n}\frac{1}{N-1}\sum_{k=3}^{5}S_{z,k} \rangle$')
ax4.legend(loc="best",prop={'size':20},ncol=1)#\langle (-1)^{n}\frac{1}{N-1}\sum_{k=1}^{N-1}S_{z,k} \rangle
#ax2.set_xscale('log')
ax4.grid()


ha = plt.figure(4,figsize=(10, 7))
ax5 = plt.axes()
ax5.plot(np.linspace(0,1,20),a[3],'b',label=r'$A_{%d,bell}$' %(4))#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
ax5.plot(np.linspace(0,1,20),b[3],'r--',label=r'$A_{%d,wgs}$' %(4))#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
ax5.plot(np.linspace(0,1,20),c[3],'g',label=r'$A_{%d,GHZ}$' %(4))#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
#ax2.plot(np.linspace(0,1,100),a[1],label='$A_2$')#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0.01$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
#ax2.plot(np.linspace(0,1,100),a[2],label='$A_3$')#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0.02$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
#ax2.plot(np.linspace(0,1,100),a[3],label='$A_4$')#,label=r'$\rm{H2I}\,\,\rm{\pi}\,\,\rm{pulse}\,\,\rm{error}\,\,e_z=0.05$')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
#ax2.plot(range(1,nt+2),j7bc300,color=colors[4],label=r'$J_z=1\,$''MHz')#label=r'$\rm{}=\rm{}\,\,$'.format('N',N))#r'$\rm{}\,\,\rm{}\,\,\rm{}=\rm{}\,\,$''MHz'.format('satellite','interaction','strength',iii))#label=r'$J=\rm{}\,\,$''MHz'.format(iii))
ax5.set_xlabel(r'$\phi$',fontsize=25)
ax5.set_ylabel(r'$\rm{SLs}$',fontsize=25)#\langle (-1)^{nT}S^{satellite}_{z} \rangle\,\, density 
ax5.tick_params(labelsize=20)#ax2.set_yticks(np.arange(-0.5, 0.6, step=0.1))
#g.xticks(size=15)# bbox_to_anchor=(0.5, 1.15)
#ax2.set_title(r'$\langle (-1)^{n}\frac{1}{N-1}\sum_{k=3}^{5}S_{z,k} \rangle$')
ax5.legend(loc="best",prop={'size':20},ncol=1)#\langle (-1)^{n}\frac{1}{N-1}\sum_{k=1}^{N-1}S_{z,k} \rangle
#ax2.set_xscale('log')
ax5.grid()

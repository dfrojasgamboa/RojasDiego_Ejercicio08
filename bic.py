import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt( 'data_to_fit.txt' )

def model_A( x, params ):
    y = params[0] + params[1] * x + params[2] * x**2
    return y

def model_B( x, params ):
    y = params[0] * np.exp( -0.5*(x-params[1])**2 / params[2]**2 )
    return y

def model_C( x, params ):
    y = params[0] * np.exp( -0.5*(x-params[1])**2 / params[2]**2 )
    y += params[0] * np.exp( -0.5*(x-params[3])**2 / params[4]**2 )
    return y

def loglike(x_obs, y_obs, sigma_y_obs, betas , model ):
    n_obs = len(y_obs)
    l = 0.0
    for i in range(n_obs):
        l += -0.5*(y_obs[i]-model(x_obs[i], betas))**2/sigma_y_obs[i]**2
    return l

def BIC( x_obs, y_obs, sigma_y_obs, betas , model , n_iterations ):
    bic = -loglike(x_obs, y_obs, sigma_y_obs, betas , model ) + 0.5 * n_pars * np.log( n_iterations )
    return 2 * bic

def run_mcmc(data, n_pars, n_iterations, model , betas ):
    x_obs = data[:, 0]
    y_obs = data[:, 1]
    sigma_y_obs = data[:, 2]

    for i in range(1, n_iterations):
        current_betas = betas[i-1,:]
        next_betas = current_betas + np.random.normal(scale=0.1, size=n_pars)

        loglike_current = loglike(x_obs, y_obs, sigma_y_obs, current_betas , model )
        loglike_next = loglike(x_obs, y_obs, sigma_y_obs, next_betas , model )

        r = np.min([np.exp(loglike_next - loglike_current), 1.0])
        alpha = np.random.random()

        if alpha < r:
            betas[i,:] = next_betas
        else:
            betas[i,:] = current_betas
    betas = betas[n_iterations//2:,:]
    return {'betas':betas, 'x_obs':x_obs, 'y_obs':y_obs}

n_dim = 1
n_iterations = 20000
n_pars = 3

betas_A = np.random.rand( n_iterations , n_pars ) - [ 12, 0 , 1 ] # np.zeros([n_iterations, n_dim+2])
resultsA = run_mcmc(data, n_pars, n_iterations , model_A , betas_A )
betasA = resultsA['betas']

plt.figure( figsize=(12,8) )
for i in range(0,n_pars):
    plt.subplot(2,2,i+1)
    plt.hist(betasA[:,i],bins=15, density=True)
    plt.title(r"$\beta_{}= {:.2f}\pm {:.2f}$".format(i,np.mean(betasA[:,i]), np.std(betasA[:,i])))
    plt.xlabel(r"$\beta_{}$".format(i))
    
X = data[:, 0]
Y = data[:, 1]
Sigma_Y = data[:, 2]
paramsA = [ np.mean(betasA[:,i]) for i in range(3) ]


plt.subplot( 2 , 2, 4)
plt.errorbar( X , Y , yerr=Sigma_Y , fmt='o' , label = 'Data' )
plt.plot( X , model_A( X , paramsA ) , 'D' , label = 'model_A' )
plt.legend()
plt.subplots_adjust(hspace=0.5)
plt.title( 'BIC = ' + str( np.round(BIC( X , Y , Sigma_Y , paramsA , model_A , n_iterations ), 5) ) )
plt.savefig("modelo_A.png",  bbox_inches='tight')    

n_pars = 3

betas_B = np.random.rand( n_iterations , n_pars ) # * 2 - 1 #- [ 1, 0 , 1 ] # np.zeros([n_iterations, n_dim+2])
resultsB = run_mcmc(data, n_pars, n_iterations , model_B , betas_B )
betasB = resultsB['betas']

plt.figure( figsize=(12,8) )
for i in range(0, n_pars):
    plt.subplot(2,2,i+1)
    plt.hist(betasB[:,i],bins=15, density=True)
    plt.title(r"$\beta_{}= {:.2f}\pm {:.2f}$".format(i,np.mean(betasB[:,i]), np.std(betasB[:,i])))
    plt.xlabel(r"$\beta_{}$".format(i))
plt.subplots_adjust(hspace=0.5)

paramsB = [ np.mean(betasB[:,i]) for i in range(3) ]

plt.subplot( 2 , 2, 4)
plt.errorbar( X , Y , yerr=Sigma_Y , fmt='o' , label = 'Data')
plt.plot( X , model_B( X , paramsB ) , 'X' , label = 'model_B')
plt.title( 'BIC = ' + str( np.round(BIC( X , Y , Sigma_Y , paramsB , model_B , n_iterations ), 5) ) )
plt.legend()
plt.savefig("modelo_B.png",  bbox_inches='tight')  

n_pars = 5

betas_C = np.random.rand( n_iterations , n_pars ) # * 2 - 1 #- [ 1, 0 , 1 ] # np.zeros([n_iterations, n_dim+2])
resultsC = run_mcmc(data, n_pars, n_iterations , model_C , betas_C )
betasC = resultsC['betas']

plt.figure( figsize=(15,10) )
for i in range(0, n_pars):
    plt.subplot(2,3,i+1)
    plt.hist(betasC[:,i],bins=15, density=True)
    plt.title(r"$\beta_{}= {:.2f}\pm {:.2f}$".format(i,np.mean(betasC[:,i]), np.std(betasC[:,i])))
    plt.xlabel(r"$\beta_{}$".format(i))
plt.subplots_adjust(hspace=0.5)

paramsC = [ np.mean(betasC[:,i]) for i in range(n_pars) ]

plt.subplot( 2 , 3, 6)
plt.errorbar( X , Y , yerr=Sigma_Y , fmt='o' , label = 'Data' )
plt.plot( X , model_C( X , paramsC ) , 'X' , label = 'model_C')
plt.legend()
plt.title( 'BIC = ' + str( np.round(BIC( X , Y , Sigma_Y , paramsC , model_C , n_iterations ), 5) ) )
plt.savefig("modelo_C.png",  bbox_inches='tight')    



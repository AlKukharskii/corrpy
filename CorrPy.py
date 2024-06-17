# CorrPy
import numpy as np

from statsmodels.tsa.stattools import acf

from scipy.stats import median_abs_deviation as mad
from scipy.ndimage import gaussian_filter1d
from scipy.special import gamma, kv
from scipy.sparse import csr_array, diags
from scipy.linalg import toeplitz
from scipy.optimize import curve_fit

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error as mse

# Definitions of used variables.

# Should add delta here.
def GrowingWindow(step, start, stop, Profiles, delta=1, method='acf', deviation='std'):
    """ 
    Description:
    This function realises the growing window method.
    
    Parameters:
    :step is a distance between two calculated profiles
    in length scale, [m].
    :start is a length scale to strat calculations, [m].
    :stop is a length scalw at which to stop, [m].
    :Profiles is a (a, b) matrix which contains studying
    profiles.
    :delta is equal to discretization. It is determined as delta = L/n
    where, L is profile's length and n is a number of points.
    :method is a switcher between the autocorrelation and
    Hurst functions.

    Return:
    :AContainer returns calculated statistical function.
    It has a shape (a, b, c) where (a) is a lag, (b) is
    a profile's length, and (c) is a profile number.
    :IWContainer returns calculated interface width.
    It has a shape (a, b) where (a) is a profile's length,
    and (b) is a profile number.
    """
    # Convert dimension of length to number of points.
    deltaL = int((stop - start)//delta)
    start = int(start//delta)
    stop = int(stop//delta)
    step = int(step//delta)

    lenP, N = np.shape(Profiles)
    # Further, I use lenP//(int(step//delta)) because lenP is given in
    # pixels when step in the length's dimension.
    # AContainer = np.full((lenP, lenP//int(step//delta), N), np.nan)
    # IWContainer = np.full((lenP//int(step//delta), N), np.nan)
    AContainer = np.full((lenP, deltaL//step + 1, N), np.nan)
    IWContainer = np.full((deltaL//step + 1, N), np.nan)
    for i in range(N):
        A, IW = loopOverProfileLength(step, start, stop,
                                      Profiles[:,i], method=method,
                                      deviation=deviation)
        
        lenA, lenStep = np.shape(A)

        AContainer[0:lenA, 0:lenStep, i] = A

        IWContainer[0:len(IW), i] = IW
    return AContainer, IWContainer

    
def loopOverProfileLength(step, start, stop, profile, method='acf', deviation='std'):
    """
    Description:
    This function iterates over profile's length-scale.

    Parameters:
    :step is a distance between two calculated profiles
    in number of points.
    :start is a length scale to strat calculations, [points].
    :stop is a length scalw at which to stop, [points].
    :profile is a studying profile.
    :method is a switcher between the autocorrelation and
    Hurst functions.

    Return:
    :AContainer returns calculated statistical function.
    :IWContainer returns calculated interface width.
    """

    if method.lower() in ['acfhuber', 'huber']:
        statFunc = acfHuber
    elif method.lower() in ['med', 'median', 'acfmed', 'acfmedian']:
        statFunc = acfMed
    elif method.lower() in ['acfnaive', 'naive']:
        statFunc = acfNaive
    else:
        statFunc = acf
        
    if deviation.lower() in ['mad']:
        devFunc = mad
    else:
        devFunc = np.std

    # in number of points
    deltaL = int(stop - start)
                 
    profile = profile[~np.isnan(profile)] #Clean out NaN's form the profiles
    AContainer = np.full((len(profile), deltaL//step + 1), np.nan)
    IWContainer = np.full((deltaL//step) + 1, np.nan)
    # for i, j in zip(range(step, len(profile), step), range(len(profile)//step)):
    for i, j in zip(range(start, stop + step, step), range(deltaL//step + 1)):
        profilePart = profile[0:i]
        # profilePart = profilePart - SVRTrend(profilePart, method="linear")
        
        A = statFunc(profilePart, nlags = len(profilePart))
        # A = acfMed(profilePart)

        IWContainer[j] = devFunc(profile[0:i])
        
        if method.lower() in ['hurst']:
            # A = np.std(profilePart)*np.sqrt(2*(1-A))
            A = IWContainer[j]*np.sqrt(2*(1-A))
        AContainer[0:len(A), j] = A
    return AContainer, IWContainer


def correlationLength(delta, A):
    """
    Description:
    This function calculates the correlation length an abscissa
    of the 1/e point for a number of autocorrelation functions.

    Parameters:
    :delta is a distance between two points in Profiles, i.e.
    disctretization.
    :A is the autocorrelation function.

    Return:
    :corrLength returns an array of the correlation length 
    values.
    """
    # Adim = A.ndim
    if A.ndim == 3:
        lenP, lenStep, N = np.shape(A)
        # lenP is profile's length.
        # lenStep is a number of profile's slices or steps.
        # N is the number of profiles.
        corrLength = np.full((lenStep, N), np.nan)
        for i in range(lenStep):
            for j in range(N):
                corrLength[i,j] = oneOverExp(delta, A[:, i, j])
        return corrLength
    elif A.ndim == 2:
        lenP, lenStep = np.shape(A)
        corrLength = np.full((lenStep), np.nan)
        for i in range(lenStep):
            corrLength[i] = oneOverExp(delta, A[:, i])
        return corrLength
    

def selfAffineParameters(delta, statFunction, method='corrLength',
                         epsilon=1.05, max_iter=1000, alpha=0.01, robust=True):
    """
    Description:
    This function estimates the self-affine exponent or
    the correlation length.

    Parameters:
    :delta is a distance between two points in Profiles.
    :statFunction is the autocorrelation function. It 
    should have a form of an one-dimensional array.
    :robust uses HuberRegressor for fitting if True; otherwise,
    it uses ordinary least squares.

    The next parameters are valid only for the robust implementation.
    :epsilon is
    :max_iter is
    :alpha is

    Return:
    :propCoeff returns an array of the proportional
    coefficients. (in dev)
    :selfAffineExp returns an array of the self-affine
    exponents.
    :corrLength returns an array of the correlation length 
    values.
    """
    
    if method.lower() in ['corrlength', 'correlation length', 'cl']:
        estimator = oneOverExp
    elif method.lower() in ['corrlengthmodel', 'correlation length model',
                            'clmodel']:
        estimator = exponentialFitCL
    elif method.lower() in ['alphamodel', 'saemodel']:
        estimator = exponentialFitSAE
    else:
        estimator = selfAffineExponent

    statFunctiondim = statFunction.ndim
    if statFunctiondim == 3:
        lenP, lenS, N = np.shape(statFunction)
        # lenP is profile's length
        # lenS is the number of slices
        # N is the number of profiles
        param = np.full((lenS, N), np.nan)
        for i in range(lenS):
            for j in range(N):
                param[i,j] = estimator(delta, statFunction[:, i, j],
                                       epsilon=epsilon, max_iter=max_iter,
                                       alpha=alpha, robust=robust)
        return param
    elif statFunctiondim == 2:
        lenP, lenS = np.shape(statFunction)
        param = np.full((lenS), np.nan)
        for i in range(lenS):
            param[i] = estimator(delta, statFunction[:, i],
                                 epsilon=epsilon, max_iter=max_iter,
                                 alpha=alpha, robust=robust)
        return param
    
def oneOverExp(delta, A, *args, **kwargs):
    """
    Description:
    This function finds the correlation length as an abscissa
    of the 1/e point of the autocorrelation function.

    Parameters:
    :delta is equal to discretization. It is determined as delta = L/n
    where, L is profile's length and n is a number of points.
    :A is the autocorrelation function. It should have a
    form of an one-dimensional array.

    Return:
    It returns a value of the correlation length.
    """
    
    loc = np.where(np.round(A, 4) == np.round(1/2.718, 4))
    if len(loc[0]) == 0:
        loc=np.where(A >= np.round(1/2.718, 4))
        if len(loc[0]) != 0:
            pos = loc[0][-1]
            x = delta*np.linspace(pos, pos+1, 2)
            xInt = delta*np.linspace(pos, pos+1, 100)
            AInt = np.interp(xInt, x, A[pos:pos+2])
            loc=np.where(AInt <= np.round(1/2.718, 4))
            return delta*(pos+((loc[0][0])/100))
        else:
            return np.nan
    else:
        return delta*loc[0][-1]

    
def selfAffineExponent(delta, H, robust=True, epsilon=1.05,
                       max_iter=1000, alpha=0.01):
    """
    Description:
    This function estimates the self-affine exponent.

    Parameters:
    :delta is equal to discretization. It is determined as delta = L/n
    where, L is profile's length and n is a number of points.
    :H is the normalised Hurst function. It should have a
    form of an one-dimensional array.
    :robust uses HuberRegressor for fitting if True; otherwise,
    it uses ordinary least squares.

    The next parameters are valid only for the robust implementation.
    :epsilon is
    :max_iter is
    :alpha is

    Return:
    It returns a value of the self-affine exponent.    
    """
    H = H[~np.isnan(H)]
    if len(H) != 0:
        pos = np.where(H >= np.sqrt(1-1/2.71))
        # pos=np.where(H >= 2/3*0.7943) # pos = np.where(H >= np.sqrt(1-1/2.71))
        H = H[1:pos[0][1]+1]
        X = delta*np.linspace(1, len(H), len(H))

        # X_train, X_test, y_train, y_test = train_test_split(X, H, shuffle=False, test_size=0.1)

        if robust is True:
            try:
                reg = HuberRegressor(epsilon=epsilon, max_iter=max_iter, alpha=alpha).fit(np.log(X.reshape(-1,1)), np.log(H))
            except ValueError:
                reg = LinearRegression(n_jobs=1).fit(np.log(X.reshape(-1,1)), np.log(H)) 
        else:
            reg = LinearRegression(n_jobs=1).fit(np.log(X.reshape(-1,1)), np.log(H))

        return reg.coef_[0]
    else:
        return np.nan

# In progress
def exponentialFitCL(delta, A, *args, **kwargs):
    """
    Description:
    This function estimates the self-affine exponent
    and correlation length based on the exponential model.

    Parameters:
    :delta is equal to discretization. It is determined as delta = L/n
    where, L is profile's length and n is a number of points.
    :A is the autocorrelation function. It should have a
    form of an one-dimensional array.

    Return:
    It returns a value of the self-affine exponent.    
    """
    # Add a cut off at large-lag error.
    # Add estimators at pre-defined alpha.
    A = A[~np.isnan(A)]
    r = delta*np.linspace(0, len(A) - 1, len(A))
    popt, pcov = curve_fit(exponentialModel, r, A)
    return popt[0]

def exponentialFitSAE(delta, A, *args, **kwargs):
    """
    Description:
    This function estimates the self-affine exponent
    and correlation length based on the exponential model.

    Parameters:
    :delta is equal to discretization. It is determined as delta = L/n
    where, L is profile's length and n is a number of points.
    :A is the autocorrelation function. It should have a
    form of an one-dimensional array.

    Return:
    It returns a value of the self-affine exponent.    
    """
    # Add a cut off at large-lag error.
    # Add estimators at pre-defined alpha.
    A = A[~np.isnan(A)]
    if len(A) != 0:
        pos=np.where(A <= 0.369) # Here should be a cut-off at large-lag error
        A = A[0:pos[0][1]+1]
        r = delta*np.linspace(0, len(A), len(A))
        try:
            popt, pcov = curve_fit(exponentialModel, r, A)
            # print(popt)
            return popt[1]
        except RuntimeError:
            return np.nan
    else:
        return np.nan
    
# Developed
    
def acfMed(profile, nlags=None):
    """
    Description:
    This function estimates the robust autocorrelation 
    function. It implements median instead of mean.

    Parameters:
    :profile is a calculated profile.

    Return:
    It returns the robust autocorrelation function.    
    """ 
    if nlags is None:
        nlags=len(profile)
    
    profile = profile - np.nanmedian(profile)
    madSqr = mad(profile, nan_policy='omit')**2
    acfContainer = np.full(len(profile), np.nan)
    for i in range(nlags):
        profileShifted = np.roll(profile, i)
        profileShifted[0:i] = np.nan
        acfContainer[i] = np.nanmedian(profile*profileShifted)/madSqr
        if acfContainer[i] < 0:
            break
    return acfContainer


def acfHuber(profile, nlags=None):
    """
    Description:
    This function estimates the robust autocorrelation 
    function. It implements Huber regression.

    Parameters:
    :profile is a calculated profile.

    Return:
    It returns the robust autocorrelation function.    
    """ 
    if nlags is None:
        nlags=len(profile)
    
    profile = profile - np.nanmedian(profile)
    profileOrig = profile
    acfContainer = np.full(len(profile), np.nan)
    HRegr = HuberRegressor(epsilon=1.01, alpha=0.0001)        
    for i in range(nlags):
        profile = profileOrig
        profileShifted = np.roll(profile, i)
        profileShifted = profileShifted[i:]
        profile = profile[i:]
        if len(profile)==0:
            break
        # X_train, X_test, y_train, y_test = train_test_split(profile, profileShifted,
        #                                             shuffle=True, test_size=0.3)
        HRegr.fit(profile.reshape(-1,1), profileShifted)
        # HRegr.fit(X_train.reshape(-1,1), y_train)
        acfContainer[i] = HRegr.coef_[-1]
        if acfContainer[i] < 0:
            break
    return acfContainer


def acfNaive(profile, nlags=None):
    """
    Description:
    This function estimates the straightforward autocorrelation 
    function.

    Parameters:
    :profile is a calculated profile.

    Return:
    It returns the robust autocorrelation function.    
    """ 
    profile = profile - np.nanmean(profile)
    madSqr = np.std(profile)**2
    acfContainer = np.full(len(profile), np.nan)
    for i in range(len(profile)):
        profileShifted = np.roll(profile, i)
        profileShifted[0:i] = np.nan
        acfContainer[i] = np.nanmean(profile*profileShifted)/madSqr
        if acfContainer[i] < 0:
            break
    return acfContainer
    

def subtractTrend(size, Profiles, method='SVR'):
    """
    Description:
    This function subtracts the profile's trend via the
    number of methods including Gaussian filter, SVRegression,
    and linear regression.

    Parameters:
    :size is the standard deviation for Gaussian filter. For 
    more details see scipy.ndimage.gaussian_filter.
    :Profiles is a (a, b) matrix which contains studying
    profiles.

    Return:
    :PWithoutTrend returns profiles without trend.
    """
    lenP, N = np.shape(Profiles)
    PWithoutTrend = np.full((lenP, N), np.nan)
    for i in range(N):
        profile = Profiles[:, i]
        profile = profile[~np.isnan(profile)]
        if method.lower() in ['gauss', 'gaussian', 'gaussian_filter']:
            profileWT = profile - gaussian_filter1d(profile, sigma=size)
        else:
            profileWT = profile - SVRTrend(profile, method)

        PWithoutTrend[0:len(profileWT), i] = profileWT
    return PWithoutTrend


def SVRTrend(profile, method="SVR"):
    """
    Description:
    This function subtracts a profile's trend with the non-linear or linear fit.

    Parameters:
    :profile is an input profile.

    Return:
    :SRegr.predict(X) returns the profile's trend.
    """
    X = np.linspace(0, len(profile) -1, len(profile)).reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, profile,
                                                        shuffle=True, test_size=0.3)
    if method.lower() in ['svr']:
        SRegr = SVR(kernel='rbf')
    else:
        SRegr = LinearRegression(n_jobs=-1)
        # SRegr = HuberRegressor()

    SRegr.fit(X_train, y_train)
    return SRegr.predict(X)

# Define sparce Toepliz matrix
def spToepliz(mainDiag, colVals, rawVals, colIndex, rawIndex, N):
    """
    Description:
    This function implements the sparce Toepliz matrix.
    """
    Vals = np.concatenate(([mainDiag], colVals, rawVals))
    Index = np.concatenate(([0], colIndex, rawIndex))
    return diags(Vals, Index, shape=(N, N))

Generator = np.random.default_rng()

def profileGenerator(alpha=0.5, omega=1, xi=10, delta=1, length=1000, N=100,
                     model='exp'):
    """ 
    Description:
    This function generates a set of random correlated profiles.
    It implements the exponential model and K-correlation model.
    Exponential model: A(r) = a*exp(-(r/xi));
    K-correlation model: A(r) = 
    
    Parameters:
    :alpha is a self-affine exponent (0 < alpha <= 1).
    :omega is a standard deviation also known as interface width.
    :xi is the characteristic length scale of exponential decay.
    :delta is a discretisation step. delta = L/n, where n is a number of points.
    :length is a profile length.
    :N is a number of profiles.

    Return:
    :Returns a set of random correlated profiles.
    """
    r = np.linspace(0, length-1, num = int(round(length/delta, 0)))
    if model.lower() in ['k', 'kcorr']:
        A = KCorrModel(r, xi, alpha)
    else:
        A = exponentialModel(r, xi, alpha)

    for index, item in enumerate(A):
        if item < 10**-5:
            A[index:] = 0
    
    # C = spToepliz(1, A[1:index], A[1:index], 
    #               np.linspace(1, index-1, index-1, dtype=np.int16), -np.linspace(1, index-1, index-1, dtype=np.int16), len(A))
    # C = toeplitz(A)
    L = csr_array(np.linalg.cholesky(toeplitz(A)))
    # L = csr_array(L)
    # return oneOverExp(delta, A), L @ np.random.normal(0, omega, size=(int(round(length/delta, 0)), N))
    return oneOverExp(delta, A), L @ Generator.normal(0, omega, size=(int(round(length/delta, 0)), N))

def exponentialModel(r, xi=10, alpha=0.5):
    '''
    Description:
    This function defines the exponential model.

    Parameters:
    :xi is the correlation length
    :alpha is a self-affine exponent.
    :r is the input grid.

    Return:
    It returns the value of the autocorrelation functions at
    points r.
    '''
    return np.exp(- (np.power(r/xi, 2*alpha)))

def KCorrModel(r, xi=10, alpha=0.5):
    '''
    Description:
    This function defines the K-correlation model.

    Parameters:
    :xi is the correlation length
    :alpha is a self-affine exponent.
    :r is the input grid.

    Return:
    It returns the value of the autocorrelation functions at
    points r.
    '''
    A = (alpha/(2**(alpha-1)*gamma(alpha+1)))*(((r/xi)*np.sqrt(2*alpha))**alpha)* \
        kv(alpha, (r/xi)*np.sqrt(2*alpha))
    A[0] = 1
    return A

def errorRel(val, refVal):
    return np.abs(val - refVal)/refVal
    
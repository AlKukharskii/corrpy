# CorrPy
import numpy as np

from numpy.lib.stride_tricks import sliding_window_view as slidingWindow

from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller

from scipy.stats import median_abs_deviation as mad
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.special import gamma, kv
from scipy.sparse import coo_array, diags, csr_array
from scipy.linalg import toeplitz
from scipy.optimize import curve_fit
from scipy.sparse.linalg import splu
from scipy.interpolate import Akima1DInterpolator

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from piecewise_regression import Fit as pw

from sksparse.cholmod import cholesky
# from sklearn.metrics import mean_squared_error as mse
# import sys

# Define global variables
# Define a constant to recalculate median absolute deviation to standard deviation
const = 1.4826 

# Generator = np.random.default_rng()

# class Analyse:
#     def __init__(self, Profiles, delta) -> None:
#         pass



# Definitions of used variables.
def loopOverProfileMWLIN(step, deltaMin, deltaMax, profile, method='acf', deviation='std'):
    # Everything comes in points
    # Moving window
    """
    Description:
    This function iterates over profile's length-scale.

    Parameters:
    :step is a distance between two calculated profiles
    in number of points.
    :start is a length scale to strat calculations, [points].
    :deltaMax is a length scalw at which to deltaMax, [points].
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

    # It should return A[lenP, deltaLen, initialPoint]
    # lenP = len(profile) (lag)
    # deltaLen = len(range(deltaMin, deltaMax, step))
    # initialPoint = lenP - deltaMin (i.e. it is a number of instances)

    # in number of points
                         
    profile = profile[~np.isnan(profile)] #Clean out NaN's form the profiles
    if deltaMin > len(profile):
        deltaMin = len(profile) - 10
        AContainer = np.full((len(profile), len(range(deltaMin, deltaMax, step)),  len(profile) - deltaMin + 1), np.nan)
        IWContainer = np.full((len(range(deltaMin, deltaMax, step)),  len(profile) - deltaMin + 1), np.nan)
        # print('Done')
        return AContainer, IWContainer
    
    AContainer = np.full((len(profile), len(range(deltaMin, deltaMax, step)),  len(profile) - deltaMin + 1), np.nan)
    IWContainer = np.full((len(range(deltaMin, deltaMax, step)),  len(profile) - deltaMin + 1), np.nan)
    # print('A.shape: ', AContainer.shape)
    # print('IW.shape: ', IWContainer.shape)

    
    for indxDelta, delta in enumerate(range(deltaMin, deltaMax, step)):
        # This loop goes over all delta.
        if delta > len(profile):
            break

        profileFrames = slidingWindow(profile, delta)
        
        profileFrames = subtractTrend(size=1, Profiles=profileFrames, method='SVR', test_size=0.5)
        
        for indxFrame, profilePart in enumerate(profileFrames):
            # This loop goes over all windows of a single profile.
            A = statFunc(profilePart, nlags = len(profilePart))   

            IWContainer[indxDelta, indxFrame] = devFunc(profilePart)
        
            if method.lower() in ['hurst']:
                A = IWContainer[indxDelta, indxFrame]*np.sqrt(2*(1-A))
                
            AContainer[0:len(A), indxDelta, indxFrame] = A
    
    return AContainer, IWContainer


def loopOverProfileMW(step, deltaMin, deltaMax, profile, method='acf', deviation='std'):
    # Everything comes in points
    # Moving window
    """
    Description:
    This function iterates over profile's length-scale.

    Parameters:
    :step is a distance between two calculated profiles
    in number of points.
    :start is a length scale to strat calculations, [points].
    :deltaMax is a length scalw at which to deltaMax, [points].
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

    # It should return A[lenP, deltaLen, initialPoint]
    # lenP = len(profile) (lag)
    # deltaLen = len(range(deltaMin, deltaMax, step))
    # initialPoint = lenP - deltaMin (i.e. it is a number of instances)

    # in number of points
                         
    profile = profile[~np.isnan(profile)] #Clean out NaN's form the profiles
    
    if deltaMin > len(profile):
        deltaMin = len(profile) - 10
        AContainer = np.full((len(profile), len(range(deltaMin, deltaMax, step)),  len(profile) - deltaMin + 1), np.nan)
        IWContainer = np.full((len(range(deltaMin, deltaMax, step)),  len(profile) - deltaMin + 1), np.nan)
        # print('Done')
        return AContainer, IWContainer
    
    AContainer = np.full((len(profile), len(range(deltaMin, deltaMax, step)),  len(profile) - deltaMin + 1), np.nan)
    IWContainer = np.full((len(range(deltaMin, deltaMax, step)),  len(profile) - deltaMin + 1), np.nan)
    # print('A.shape: ', AContainer.shape)
    # print('IW.shape: ', IWContainer.shape)

    
    for indxDelta, delta in enumerate(range(deltaMin, deltaMax, step)):
        # This loop goes over all delta.
        if delta > len(profile):
            break

        profileFrames = slidingWindow(profile, delta)
        # print(profileFrames.shape)
        
        for indxFrame, profilePart in enumerate(profileFrames):
            # This loop goes over all windows of a single profile.
            A = statFunc(profilePart, nlags = len(profilePart))   

            IWContainer[indxDelta, indxFrame] = devFunc(profilePart)
        
            if method.lower() in ['hurst']:
                A = IWContainer[indxDelta, indxFrame]*np.sqrt(2*(1-A))
                
            AContainer[0:len(A), indxDelta, indxFrame] = A
    
    return AContainer, IWContainer



def MWSingle(Profiles, step, delta):
    step /= delta
    step = int(step)
    lengthIn = step + 1

    l, N = Profiles.shape

    xiLst = np.full((l, N), np.nan)
    iwLst = np.full((l, N), np.nan)
    # alphaLst = np.full((5000, 90), np.nan)
        
    for i in range(N): # Iterate over profiles
        A, IW = loopOverProfileMW(step=1, deltaMin=step, deltaMax=lengthIn, profile=Profiles[:, i], method='acf', deviation='mad') #Input in points
    
        if not np.sum(~np.isnan(A)):
            continue

        # H = np.sqrt(1 - A)
        try:
            xi = correlationLength(delta, A)
            # alpha = selfAffineParameters(delta, np.sqrt(1 - A), 'exponent', robust=False, cut_off=0.5)
            xiLst[0:xi.shape[1], i] = xi
            iwLst[0:xi.shape[1], i] = IW
        except:
            pass
            # xi = xi#np.full([2, 1], np.nan)

        # xiLst[0:xi.shape[1], i] = xi
        # iwLst[0:xi.shape[1], i] = IW
        # alphaLst[0:xi.shape[1], i] = alpha

    else:
        xiMed = np.nanmedian(xiLst,1)
        xiMAD = const*mad(xiLst, 1, nan_policy='omit')

        iwMed = np.nanmedian(iwLst,1)
        iwMAD = const*mad(iwLst, 1, nan_policy='omit')


        x = delta*np.linspace(0, xiMed.shape[0]-1, xiMed.shape[0]) + (step*delta)


        BMed = step*delta/xiMed

        # Calculate errors
        sysXiError = systematicErrorXi(BMed, xiMed)
        xiErr = fullError(sysXiError, const*xiMAD/np.sqrt(Profiles.shape[1]))
    return x, xiMed, xiErr, xiMAD, iwMed, iwMAD


def MovingWindow(step, deltaMin, deltaMax, Profiles, delta=1, method='acf', deviation='std'):
    """ 
    Description:
    This function realises the growing window method.
    
    Parameters:
    :step is a distance between two calculated profiles
    in length scale, [m].
    :deltaMin is a length scale to strat calculations, [m].
    :deltaMax is a length scalw at which to deltaMax, [m].
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
    # It should return A[lenP, deltaLen, InitialPoint, N]
    # lenP = len(profile) (lag)
    # deltaLen = len(range(deltaMin, deltaMax, step))
    # initialPoint = lenP - deltaMin (i.e. it is a number of instances)
    # N is a number of profiles

    # Convert dimension of length to number of points.
    deltaMin = int(np.ceil(deltaMin/delta))
    deltaMax = int(np.ceil(deltaMax/delta))
    step = int(np.ceil(step/delta))

    lenP, N = np.shape(Profiles)
    # Further, I use lenP//(int(step//delta)) because lenP is given in
    # pixels when step in the length's dimension.

    AContainer = np.full((lenP, len(range(deltaMin, deltaMax, step)),  lenP - deltaMin + 1, N), np.nan)
    IWContainer = np.full((len(range(deltaMin, deltaMax, step)),  lenP - deltaMin + 1, N), np.nan)
    for i in range(N):
        # This loop goes over all profiles.
        A, IW = loopOverProfileMW(step, deltaMin, deltaMax,
                                      Profiles[:,i], method=method,
                                      deviation=deviation)
        
        lenA, lenStep, lenP = A.shape
        
        AContainer[0:lenA, 0:lenStep, 0:lenP, i] = A

        IWContainer[0:lenStep, 0:lenP, i] = IW
        
    return AContainer, IWContainer


def GrowingWindow(step, start, stop, Profiles, delta=1, method='acf', deviation='std',
                  reduced=False):
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
    if reduced:
        iterator = loopOverProfileMW
    else:
        iterator = loopOverProfileLength

    # Convert dimension of length to number of points.
    deltaL = int(np.ceil(stop - start)/delta)
    start = int(np.ceil(start/delta))
    stop = int(np.ceil(stop/delta))
    step = int(np.ceil(step/delta))

    lenP, N = np.shape(Profiles)
    # Further, I use lenP//(int(step//delta)) because lenP is given in
    # pixels when step in the length's dimension.
    # AContainer = np.full((lenP, lenP//int(step//delta), N), np.nan)
    # IWContainer = np.full((lenP//int(step//delta), N), np.nan)
    AContainer = np.full((lenP, int(np.ceil(deltaL/step))+1, N), np.nan)
    IWContainer = np.full((int(np.ceil(deltaL/step))+1, N), np.nan)
    for i in range(N):
        # A, IW = loopOverProfileLength(step, start, stop,
        #                               Profiles[:,i], method=method,
        #                               deviation=deviation)
        A, IW = iterator(step, start, stop,
                                      Profiles[:,i], method=method,
                                      deviation=deviation)
        
        if reduced:
            A = np.nanmedian(A, -1)
            IW = np.nanmedian(IW, -1)
        
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
    
    AContainer = np.full((len(profile), int(np.ceil(deltaL/step))+1), np.nan)
    IWContainer = np.full(int(np.ceil(deltaL/step))+1, np.nan)
    # for i, j in zip(range(step, len(profile), step), range(len(profile)//step)):
    for i, j in zip(range(start, stop + step, step), range(int(np.ceil(deltaL/step)) + 1)):
        if i > len(profile):
            break

        profilePart = profile[0:i]
        
        # profilePart = profilePart - SVRTrend(profilePart, method="linear")
        
        A = statFunc(profilePart, nlags = len(profilePart))
        # A = acfMed(profilePart)

        # IWContainer[j] = devFunc(profile[0:i])
        IWContainer[j] = devFunc(profilePart)
        
        if method.lower() in ['hurst']:
            # A = np.std(profilePart)*np.sqrt(2*(1-A))
            A = IWContainer[j]*np.sqrt(2*(1-A))

        AContainer[0:len(A), j] = A
    return AContainer, IWContainer


def correlationLength(delta, A, *args, **kwargs):
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
    

def selfAffineParameters(delta, statFunction, method='corrLength', *args, **kwargs):
                         #epsilon=1.05, max_iter=1000, alpha=0.01, robust=True):
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
                # param[i,j] = estimator(delta, statFunction[:, i, j],
                #                        epsilon=epsilon, max_iter=max_iter,
                #                        alpha=alpha, robust=robust)
                param[i,j] = estimator(delta, statFunction[:, i, j], *args, **kwargs)
        return param
    elif statFunctiondim == 2:
        lenP, lenS = np.shape(statFunction)
        param = np.full((lenS), np.nan)
        for i in range(lenS):
            # param[i] = estimator(delta, statFunction[:, i],
            #                      epsilon=epsilon, max_iter=max_iter,
            #                      alpha=alpha, robust=robust)
            param[i] = estimator(delta, statFunction[:, i], *args, **kwargs)
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
    nInterp = 200
    
    # nFitA = 3
    
    # A = A[~np.isnan(A)]
    # X = delta*np.linspace(0, len(A) - 1, len(A))
    # X, A = fitData(delta, X, A, interpPoints=int(nFitA*len(A)))
    # delta = delta/nFitA
    
    loc = np.where(np.round(A, 4) == np.round(1/2.718, 4))
    if len(loc[0]) == 0:
        loc=np.where(A >= np.round(1/2.718, 4))
        if len(loc[0]) != 0:
            pos = loc[0][-1]
            x = delta*np.linspace(pos, pos+1, 2)
            xInt = delta*np.linspace(pos, pos+1, nInterp)
            AInt = np.interp(xInt, x, A[pos:pos+2])
            loc=np.where(AInt <= np.round(1/2.718, 4))
            return delta*(pos+((loc[0][0])/nInterp))
        else:
            return np.nan
    else:
        return delta*loc[0][-1]

def selfAffineExponent(delta, H, *args, **kwargs):
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
        X = delta*np.linspace(0, len(H) - 1, len(H))
        # X, H = fitData(delta, X, H, interpPoints=4*len(H))
        # X, H = fitData(delta, X, H, interpPoints=2*len(H))
        
        pos=np.where(H >= .8*0.7943)
        # pos=np.where(H >= .85*0.7943)
        
        H = H[0:pos[0][1]]
        
        X = delta*np.linspace(0, len(H) - 1, len(H))
        
        # X, X_test, H, y_test = train_test_split(X, H,shuffle=True,
        #                                         test_size=0.2)

        popt, pcov = curve_fit(powerlaw, X, H, (0.1, 0.5))
        return popt[1]/correctionFactor(0.8)
    else:
        return np.nan

# def selfAffineExponent(delta, H, robust=True, epsilon=1.05,
#                        max_iter=1000, alpha=0.01, cut_off=0.45, *args, **kwargs):
#     """
#     Description:
#     This function estimates the self-affine exponent.

#     Parameters:
#     :delta is equal to discretization. It is determined as delta = L/n
#     where, L is profile's length and n is a number of points.
#     :H is the normalised Hurst function. It should have a
#     form of an one-dimensional array.
#     :robust uses HuberRegressor for fitting if True; otherwise,
#     it uses ordinary least squares.

#     The next parameters are valid only for the robust implementation.
#     :epsilon is
#     :max_iter is
#     :alpha is

#     Return:
#     It returns a value of the self-affine exponent.    
#     """
#     H = H[~np.isnan(H)]
#     if len(H) != 0:
#         # Find where to cut the curve and cut it.
#         pos = np.where(H >= cut_off*0.7943) # pos = np.where(H >= np.sqrt(1-1/2.71))
#         H = H[0:pos[0][1]]
#         # Generate the length scale
#         X = delta*np.linspace(0, len(H), len(H))
    
#         # X, X_test, H, y_test = train_test_split(X, H,shuffle=True,
#         #                                         test_size=0.2)

#         popt, pcov = curve_fit(powerlaw, X, H, (0.1, 0.5))
#         # popt, pcov = curve_fit(exponentialModel, X, H, (0.1, 0.5))
#         if pcov[0][0] == np.inf:
#             return np.nan
#         else:
#             return popt[1]

#         # The next part is under development. Robust fitting of alpha.

#         # if robust is True:
#         #     try:
#         #         reg = HuberRegressor(epsilon=epsilon, max_iter=max_iter, alpha=alpha, fit_intercept=True).fit(np.log(X.reshape(-1,1)), np.log(H))
#         #     except ValueError:
#         #         reg = LinearRegression(n_jobs=1, fit_intercept=True).fit(np.log(X.reshape(-1,1)), np.log(H)) 
#         # else:
#         #     reg = LinearRegression(n_jobs=1, fit_intercept=True).fit(np.log(X.reshape(-1,1)), np.log(H))

#         # return reg.coef_[0]
#     else:
#         return np.nan
    
def fitData(delta, X, Y, interpPoints=100):
    """
    A supporting function for fast fitting.
    """
    XInt = delta*np.linspace(0, len(Y), interpPoints)
    YInt = Akima1DInterpolator(X, Y, method="makima")(XInt)
    # YInt = np.interp(XInt, X, Y) #Linear interpolation
    return XInt, YInt

def powerlaw(x, a, b):
    """
    Fit a power-law.
    """
    return a*pow(x,b)

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
def Range(profile):
    return np.max(profile) - np.min(profile)
    
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
    

def subtractTrend(size, Profiles, method='SVR', test_size=0.7):
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
    :test_size sets the size of the train-test split.

    Return:
    :PWithoutTrend returns profiles without trend.
    """
    lenP, N = np.shape(Profiles)
    PWithoutTrend = np.full((lenP, N), np.nan)
    for i in range(N):
        profile = Profiles[:, i]
        profile = profile[~np.isnan(profile)]
        if method.lower() in ['gauss', 'gaussian', 'gaussian_filter']:
            profileWT = profile - gaussian_filter1d(profile, sigma=size, mode='nearest')
        elif method.lower() in ['med', 'median', 'median_filter']:
            profileWT = profile - median_filter(profile, size=size, mode='nearest')
        else:
            profileWT = profile - SVRTrend(profile, method, test_size)

        PWithoutTrend[0:len(profileWT), i] = profileWT
    return PWithoutTrend


def SVRTrend(profile, method="SVR", test_size=0.7):
    """
    Description:
    This function subtracts a profile's trend with the non-linear or linear fit.

    Parameters:
    :profile is an input profile.
    :method sets the used method. Choose between SVR and linear regression (OLS).
    :test_size sets the size of the train-test split.

    Return:
    :SRegr.predict(X) returns the profile's trend.
    """
    X = np.linspace(0, len(profile) -1, len(profile))#.reshape(-1, 1)
    # X = np.linspace(0, len(profile) -1, len(profile)).reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, profile,
                                                        shuffle=True, test_size=test_size)

    if method.lower() in ['svr']:
        SRegr = SVR(kernel='rbf')
        # SRegr = LinearRegression(n_jobs=-1)
        SRegr.fit(X_train.reshape(-1, 1), y_train)
        y = SRegr.predict(X.reshape(-1, 1))

    elif method.lower() in ['linear', 'lin']:
        SRegr = LinearRegression(n_jobs=-1)
        SRegr.fit(X_train.reshape(-1, 1), y_train)
        y = SRegr.predict(X.reshape(-1, 1))
        
    else:
        # BIC = model_selection(X_train, X_test, y_train, y_test, range(1, 4))
        # n = min(BIC, key=BIC.get)
        n = 1
        SRegr = pw(X_train, y_train, n_breakpoints=n if n.is_integer() else 1)
        y = SRegr.predict(X)

        try:
            _ = profile - y
        except:
            SRegr = SVR(kernel='rbf', C=0.1)
            SRegr.fit(X_train.reshape(-1, 1), y_train)
            y = SRegr.predict(X.reshape(-1, 1))

        # SRegr = LinearRegression(n_jobs=-1)
        # SRegr = HuberRegressor()

    # SRegr.fit(X_train, y_train)
    return y


def bayesian_information_criterion(x, std, n_breakpoints):
        """
        Calculates the Bayesian Information Criterion of a piecewise
        linear model.
        """
        n = len(x)  # No. data points
        k = 2 + 2 * n_breakpoints  # No. model parameters
        return n * np.log(std**2) + k * np.log(n)


def model_selection(X_train, X_test, y_train, y_test, param):
    BIC = {}

    for n in param:
        # print('n: ', n)
        try:
            pw_fit = pw(X_train, y_train, n_breakpoints=n)

            y = pw_fit.predict(X_test)
            error = np.abs(y - y_test)
            std = const*mad(error)
            bic = bayesian_information_criterion(X_test, std, n)
        except:
            bic = np.inf

        BIC[n] = bic
    return BIC

# Define sparce Toepliz matrix
def spToepliz(mainDiag, colVals, rawVals, colIndex, rawIndex, N):
    """
    Description:
    This function implements the sparce Toepliz matrix.
    """
    Vals = np.concatenate(([mainDiag], colVals, rawVals))
    Index = np.concatenate(([0], colIndex, rawIndex))
    return diags(Vals, Index, shape=(N, N), format='csc')
 

# Generator = np.random.default_rng()

def profileGenerator(alpha=0.5, omega=1, xi=10, delta=1, length=1000, N=100,
                     model='exp', randomTrend=False, splitPos=0.5, coeffWidth=12.5, numberOfSplits=2):
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
    Generator = np.random.default_rng()

    r = np.linspace(0, length-1, num = int(round(length/delta, 0)))
    if model.lower() in ['k', 'kcorr']:
        A = KCorrModel(r, xi, alpha)
    else:
        A = exponentialModel(r, xi, alpha)

    # Need to use scisparse
    lenA = len(A)

    for index, item in enumerate(A):
        if item < 10**-4:
            A[index:] = 0
            break

    Aindx = A[1:index]
    A = coo_array(A)
    indexs = np.linspace(1, index-1, index-1, dtype=np.int16)
    C = spToepliz(1, Aindx, Aindx, indexs, -indexs, lenA)
    L = cholesky(C, ordering_method='natural').L()
    Profiles = L @ Generator.normal(0, omega, size=(int(round(length/delta, 0)), N))

    # Do not need to use scisparse
    # L = csr_array(np.linalg.cholesky(toeplitz(A)))
    # Profiles = oneOverExp(delta, A), L @ Generator.normal(0, omega, size=(int(round(length/delta, 0)), N))
    
    if randomTrend:
        Profiles = addRandomTrend(Profiles, r, delta, length, N, splitPos, coeffWidth, numberOfSplits)
    return oneOverExp(delta, np.insert(Aindx,0,1)), Profiles

def addRandomTrend(Profiles, r, delta=1, length=1000, N=100, splitPos=0.5, coeffWidth=12.5, numberOfSplits=2):
    """
    Description:
    This function add random linear trends to the given profiles.

    Parameters:
    :Profiles

    Return:
    It returns profiles with added random linear trends.
    """
    Generator = np.random.default_rng()
    # splitPos = np.linspace()
    # splitPoints = 

    if numberOfSplits == 2:
        # Generate random splits
        splitPoints = np.array([*map(int, Generator.normal(splitPos*length, length/50, size=(N, numberOfSplits-1))/delta)])
        # Generate random coefs
        coeff = Generator.normal(0, coeffWidth/length, size=(numberOfSplits, N))

        for k1, k2, n, i in zip(coeff[0,:], coeff[1,:], splitPoints, range(0, N)):
            y = k1*r
            # y[n:] = -k1*r[n:] + 2*k1*r[n]
            y[n:] = -k2*r[n:] + (k1*r[n] + k2*r[n])
            Profiles[:, i] = Profiles[:, i] + y
    
    return Profiles


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
    """
    Description:
    Calculates the relative error between val and refVal.
    """
    return np.abs(val - refVal)/refVal
    
def systematicErrorXi(B, xi):
    """
    Description:
    This function estimates the systematic error of the xi's estimator.

    Parameters:
    :B is an independent number of points in a correlated system.
    :xi is the value of the lateral correlation length.

    Return:
    It returns absolute value of the systematic error.
    """
    return xi*10/6*(np.sqrt(1/B) - 0.05)

def systematicErrorAlpha(B, alpha):
    """
    Description:
    This function estimates the systematic error of the alpha's estimator.

    Parameters:
    :B is an independent number of points in a correlated system.
    :alpha is the value of the self-affine exponent.

    Return:
    It returns absolute value of the systematic error.
    """
    return alpha*10/53*(np.sqrt(1/B) - 0.06)

def fullError(systematicError, randomError):
    """
    Description:
    This function estimates the full error.

    Parameters:
    :systematicError is the systematic error of an estimator.
    :randomError is the random error of an estimator.

    Return:
    It returns the full error of the estimator.
    """
    return np.sqrt(np.power(systematicError, 2) + np.power(randomError, 2))

def correctionFactor(psi=0.8):
    return -0.187*psi**2.312 + 1

# def test(*args, **kwargs):
#     x = np.linspace(0,5000,5000)
#     LengthLst = range(20, 120, 20)

#     # Add alpha

#     XI = []
#     XIMAD = []
#     ALPHA = []
#     ALPHAMAD = []

#     for step in LengthLst:
#         # print(step)
#         lengthIn = step + 1 

#         xiLst = np.full((5000, N), np.nan)
#         # alphaLst = np.full((5000, N), np.nan)
        
#         for i in range(0, N): # Iterate over profiles
#             # A, IW = cp.loopOverProfileMW(step=1, deltaMin=step, deltaMax=lengthIn, profile=PWT[:, i], method='acf', deviation='std')
#             A, IW = cp.loopOverProfileMWLIN(step=1, deltaMin=step, deltaMax=lengthIn, profile=PWT[:, i], method='acf', deviation='std')
            
#             if not np.sum(~np.isnan(A)):
#                 # print('Profile length is less than delta.')
#                 continue

#             # H = np.sqrt(1 - A)

#             xi = cp.correlationLength(deltaIn, A)
#             # alpha = cp.selfAffineParameters(deltaIn, np.sqrt(1 - A), 'exponent', robust=False, cut_off=0.5)

#             xiLst[0:xi.shape[1], i] = xi
#             # alphaLst[0:xi.shape[1], i] = alpha

#         else:
#             xiMed = np.nanmedian(xiLst,1)
#             xiMAD = const*cp.mad(xiLst, 1, nan_policy='omit')

#             # alphaMed = np.nanmedian(alphaLst,1)
#             # alphaMAD = const*cp.mad(alphaLst, 1, nan_policy='omit')

#         XI.append(np.nanmedian(xiMed))
#         XIMAD.append(np.nanmedian(xiMAD))

#         # ALPHA.append(np.nanmedian(alphaMed))
#         # ALPHAMAD.append(np.nanmedian(alphaMAD))

def diffLoop(Profiles):
    Profiles_diff = np.full_like(Profiles, np.nan)
    for indx in range(Profiles.shape[1]):
        profile = Profiles[:, indx]
        profile = profile[~np.isnan(profile)]
        res = np.diff(profile)
        Profiles_diff[0 : len(res), indx] = res
    return Profiles_diff

def testAdfuller(Profiles):
    pValLst = []
    critVal = []
    for indx in range(Profiles.shape[1]):
        profile = Profiles[:, indx]
        profile = profile[~np.isnan(profile)]
        res = adfuller(profile, regression='ct')
        pValLst.append(res[1])
        critVal.append(True if res[0] < res[4]['1%'] else False)
    return pValLst, critVal
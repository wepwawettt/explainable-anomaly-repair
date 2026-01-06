from statsmodels.tsa.seasonal import STL

def stl_decompose(series, period):
    stl = STL(series, period=period, robust=True)
    result = stl.fit()
    return result.trend, result.seasonal, result.resid

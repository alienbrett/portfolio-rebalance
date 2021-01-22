import numpy as np
import pandas as pd

__all__ = ['optimal_portfolio', 'rebalance']



def add_derived(df):
    '''Given price and share columns, derive other useful rows in one go
    '''
    df['value'] = df['px'] * df['shares']

    # Sum of value of portfolio
    total_value = df['value'].sum()

    # Current allocation of each item
    df['alloc'] = df['value'] / total_value

    # How much would allocation change after buying one share?
    # Notice that '$' is wrong, but this wont' affect the algorithm
    df['d_alloc'] = df['px'] / total_value


def test_all(df, error):
    '''Return two lists, the new utility 
    '''
    e = error(df['target'] - df['alloc'])
    plus = [e]
    minus = [e]
    
    # Set up some memory in advance
    p_alloc = df['alloc'].values
    target = df['target'].values
    vec = 0 * p_alloc
    d_alloc = df['d_alloc'].values
    x0 = target - p_alloc
    
    for i in range(1,len(df)):

        # Generate difference vector
        # This will be summed appropriately
        vec = 0 * p_alloc
        vec[0] = -d_alloc[i]
        vec[i] = d_alloc[i]

        # Store back the values
        plus.append(
            error( x0 - vec )
        )
        minus.append (
            error( x0 + vec )
        )
        
    return plus, minus



def pick_move(p,n):
    '''Given a set of hypothetical error, spit out ('buy', n)
    for a row number n, and 'buy' or 'sell' depending on move
    Otherwise None for done
    '''
    e = p[0]
    m_p = np.argmin(p)
    m_n = np.argmin(n)
    
    if p[m_p] < np.min( [e, n[m_n]] ):
        # Lowers utility to purchase
        return 'buy', m_p
    elif n[m_n] < np.min( [e, p[m_p]] ):
        # Lowers utility to sell
        return 'sell', m_n
    else:
        return None
    
    
def move_one(df, error):
    
    p, n = test_all(df, error=error)
    
    move = pick_move(p, n)
    
    if move is None:
        return None
    
    which, row = move
    buysell = -1 if which == 'sell' else 1

    # Add d_alloc into the proper row and cash
    dx = df.copy()
    
    # Increment shares
    dx['shares'].iloc[row] += buysell
    dx['shares'].iloc[0]   -= buysell * dx['px'].iloc[row]
    
    add_derived(dx)
    return dx
    

    

def optimal_portfolio( px, target, market_value, pct_cash, verbose=False, error=np.linalg.norm ):
    '''Finds portfolio of integer share counts that best approximates a desired allocation, subject to custom error function.
    
    Arguments:
        px: array-like, price of security
        target: array-like, desired allocation excluding cash, must sum to 1
        market_value: float, value of entire portfolio
        pct_cash: float, desired cash allocation
        verbose: whether to display convergence info (default False)
        error: error function to use, defaults to least squares
    
    Returns:
        (cash, share_delta) where cash is float, and share_delta array-like
    '''
    #### Make an educated guess as to the correct allocations
    # Append unit currency
    px = np.array([1] + list(px))
    # Adjust target allocations to account for cash
    target *= (1-pct_cash)
    target = np.array([pct_cash] + list(target))
    
    # Guestimate shares, based on float multiplication
    shares = np.round(market_value * target / px)
    # Imply cash allocation
    shares[0] = market_value - np.sum((shares * px)[1:])
    
    # Create dataframe that will be base unit of operations
    dx = pd.DataFrame(
        {
            'px':     px,
            'shares': shares,
            'target': target
        },
    )
    add_derived(dx)
    
    # Now optimize to better solution
    flag = True
    while flag:
        dxx = move_one(dx, error=error)
        flag = dxx is not None

        if flag:
            if verbose:
                print( 'Change in shares:')
                print( (dxx['shares'] - dx['shares']).iloc[1:] )
                print()
            dx = dxx
    if verbose:
        print(n, 'steps taken')

    return (dx['shares'][0], dx['shares'].values[1:])



def rebalance( px, target, market_value, pct_cash, shares, cash, verbose=False, error=np.linalg.norm ):
    '''Returns the change in shares and cash needed to optimally balance allocation, according to 'optimal_portfolio()'
    Arguments:
        px: array-like, price of security
        target: array-like, desired allocation excluding cash, must sum to 1
        market_value: float, value of entire portfolio
        pct_cash: float, desired cash allocation
        shares: array-like, currently-held shares
        cash: float, currently-held cash
        verbose: whether to display convergence info (default False)
        error: error function to use, defaults to least squares
    
    Returns:
        (cash, share_delta) where cash is float, and share_delta array-like
    '''
    cash_held, alloc = optimal_portfolio(
        px = px,
        target = target,
        market_value = market_value,
        pct_cash = pct_cash,
        verbose=verbose, 
        error=error
    )
    
    return (
        cash_held - cash,
        alloc - shares
    )
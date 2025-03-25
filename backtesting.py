capital = 1_000_000
com = 0.125/100
n_shares = 2_000

portfolio_value = [capital]

stop_loss = 0.08328943650714873
take_profit = 0.1052374295703637

wins = 0
losses = 0

active_long_pos = None
active_short_pos = None

for i, row in dataset.iterrows():
    #Close Long position
    if active_long_pos:
        # Closed by stop loss
        if row.Close < active_long_pos['stop_loss']:
            pnl = row.Close * n_shares * (1-com)
            capital += pnl
            active_long_pos = None
    if active_long_pos:
        # Closed by take profit
        if row.Close > active_long_pos['take_profit']:
            pnl = row.Close * n_shares * (1-com)
            capital += pnl
            active_long_pos = None 
    # Close Short Poisitions

    # Open Long Pos
    if row.RSI_BUY and active_long_pos is None:
        cost = row.Close * n_shares * (1+com)

        if capital > cost:
            capital -= cost

            active_long_pos = {
                'datetime': row.Datetime,
                'opened_at': row.Close,
                'take_profit': row.Close * (1+take_profit),
                'stop_loss': row.Close * (1-stop_loss)
            }

    # Open short pos

    # Calculate Long positions value
    long_value = 0
    if active_long_pos:
        long_value = row.Close * n_shares

    # Calculate port value
    portfolio_value.append(capital + long_value)
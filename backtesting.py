import pandas as pd

def run_backtest(dataset, initial_capital=1_000_000, n_shares=2000, com=0.125 / 100,
                 stop_loss=0.08328943650714873, take_profit=0.1052374295703637,
                 verbose=False):

    capital = initial_capital
    portfolio_value = [capital]

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
        if active_short_pos:
            if row.Close > active_short_pos['stop_loss']:
                # recomprar caro = pérdida
                cost = row.Close * n_shares * (1 + com)
                pnl = active_short_pos['entry'] * n_shares - cost
                capital += active_short_pos['entry'] * n_shares - pnl  # restamos pérdida
                active_short_pos = None
            elif row.Close < active_short_pos['take_profit']:
                # recomprar barato = ganancia
                pnl = active_short_pos['entry'] * n_shares - row.Close * n_shares * (1 + com)
                capital += active_short_pos['entry'] * n_shares + pnl
                active_short_pos = None

        # Open Long Pos
        if row.predicted_signal == 'BUY' and active_long_pos is None:
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
        if row.predicted_signal == 'SELL' and active_short_pos is None and active_long_pos is None:
            proceeds = row.Close * n_shares * (1 - com)
            capital += proceeds
            active_short_pos = {
                'datetime': row.name,
                'entry': row.Close,
                'take_profit': row.Close * (1 - take_profit),
                'stop_loss': row.Close * (1 + stop_loss)
            }
        # Calculate Long positions value
        position_value = 0
        if active_long_pos:
            long_value = row.Close * n_shares
        elif active_short_pos:
            position_value = active_short_pos['entry'] * n_shares - row.Close * n_shares

        # Calculate port value
        portfolio_value.append(capital + position_value)
    
    return portfolio_value, capital
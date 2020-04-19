import pickle

# Getting back the results:
with open('fqiETResults.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    myFleets, myAgent, G, Q, batch_t, batch_xt, batch_at, batch_rt1, batch_t1, batch_xt1, fig_1 = pickle.load(f)

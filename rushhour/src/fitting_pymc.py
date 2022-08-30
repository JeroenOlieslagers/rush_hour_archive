from pymc import MCMC
import lrta_pymc as model

iter_=200
burn_=100
thin_=1

M = MCMC(model,db='pickle', dbname='lrta.pickle')
M.sample(iter=iter_, burn=burn_, thin=thin_)
M.db.close()
for s in model.sample_data.iteritems():
    print s

# TODO: Implement LOOCV: sum([log(sum([w_s* p(h(i)|s) for s in samples])/all_w) for i in instance])
# TODO: Check convergence of samples
# TODO: Tune and improve performance






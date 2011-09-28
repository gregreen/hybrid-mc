#include <iostream>
#include <vector>
#include <math.h>

#include "hybridmc.h"
#include "binner.h"
#include "stats.h"

#define sqr(x) ((x)*(x))

using namespace std;


struct TSHO {
	vector<double> q_0;
	vector<double> w2;
	size_t dim;
	double weight_1, weight_2;
	
	TSHO() { dim = 0; weight_1 = 0.5; weight_2 = 0.5; }
	
	void add_oscillator(double _q_0, double w) {
		q_0.push_back(_q_0);
		w2.push_back(w*w);
		dim++;
	}
};

double negE_SHO(const double *const q, size_t dim, TSHO &params) {
	double p = 0.;
	double tmp = 0.;
	for(size_t i=0; i<dim; i++) { tmp += 0.5*params.w2[i]*sqr(q[i]-params.q_0[i]); }
	p += params.weight_1 * exp(-tmp);
	tmp = 0;
	for(size_t i=0; i<dim; i++) { tmp += 0.5*params.w2[i]*sqr(q[i]+params.q_0[i]); }	// Mirror potential
	p += params.weight_2 * exp(-tmp);
	return log(p);
}

double rand_state(double *q, size_t dim, gsl_rng *r, TSHO &params) {
	for(size_t i=0; i<dim; i++) { q[i] = gsl_ran_gaussian_ziggurat(r, 1.); }
}

int main(int argc, char **argv) {
	unsigned int N_threads = 7;
	unsigned int N_burn_in = 1000;
	unsigned int N_steps = 5000;
	unsigned int N_rounds = 100;
	double eta = 0.02;
	unsigned int L = 20;
	double target_acceptance = 0.98;
	double convergence_threshold = 1.025;
	string outfile = "bins.txt";
	
	TSHO sho;
	
	sho.add_oscillator(2., 1./1.);
	sho.add_oscillator(3., 1./1.);
	sho.add_oscillator(2., 1./1.);
	sho.weight_1 = 0.5;
	sho.weight_2 = 0.5;
	
	double min[3] = {-10, -10, -10};
	double max[3] = {10, 10, 10};
	unsigned int width[3] = {100, 100, 100};
	TBinnerND logger(&min[0], &max[0], &width[0], 3);
	TStats stats(3);
	
	/*THybridMC<TSHO, TBinnerND> hmc(sho.dim, negE_SHO, rand_state, sho, logger, stats);
	double q_0[3] = {1., 2., 3.};
	double p_0[3] = {0., 0., 0.};
	hmc.test_integration(&q_0[0], &p_0[0], 10000, 0.001);*/
	
	TParallelHybridMC<TSHO, TBinnerND> hmc(N_threads, sho.dim, negE_SHO, rand_state, sho, logger, stats);
	
	hmc.tune(L, eta, target_acceptance, 50);
	cout << endl << "eta -> " << eta << endl;
	cout << "Empirical acceptance rate = " << hmc.acceptance_rate() << "\t(target = " << target_acceptance << ")" << endl << endl;
	hmc.clear_acceptance_rate();
	
	hmc.step_multiple(N_burn_in, L, eta, false);
	//hmc.update_stats();
	cout << "Burn-in acceptance rate: " << hmc.acceptance_rate() << endl << endl;
	
	hmc.tune(L, eta, target_acceptance, 50);
	cout << "eta -> " << eta << endl;
	cout << "Empirical acceptance rate = " << hmc.acceptance_rate() << "\t(target = " << target_acceptance << ")" << endl << endl;
	hmc.clear_acceptance_rate();
	
	bool converged;
	double GR_i;
	for(unsigned int n=0; n<N_rounds; n++) {
		hmc.step_multiple(N_steps, L, eta, true);
		hmc.update_stats();
		hmc.calc_GR_stat();
		cout << "===========================================================" << endl;
		cout << "|| n = " << n+1 << endl;
		cout << "===========================================================" << endl << endl;
		cout << "Gelman-Rubin diagnostic:";
		converged = true;
		for(unsigned int i=0; i<3; i++) {
			GR_i = hmc.get_GR_stat(i);
			cout << " " << setprecision(5) << GR_i; 
			if(GR_i > convergence_threshold) { converged = false; }
		}
		cout << endl << "Acceptance rate: " <<  hmc.acceptance_rate() << endl;
		stats.print();
		cout << endl << endl;
		if(converged) { break; }
	}
	
	logger.write_to_file(outfile, true, true);
	
	return 0;
}

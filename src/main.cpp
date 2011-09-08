#include <iostream>
#include <vector>
#include <math.h>

#include "hybridmc.h"

#define sqr(x) ((x)*(x))

using namespace std;


struct TSHO {
	vector<double> q_0;
	vector<double> w2;
	size_t dim;
	
	TSHO() { dim = 0; }
	
	void add_oscillator(double _q_0, double w) {
		q_0.push_back(_q_0);
		w2.push_back(w*w);
		dim++;
	}
};

double negE_SHO(const double *const q, size_t dim, TSHO &params) {
	double E = 0.;
	for(size_t i=0; i<dim; i++) { E += 0.5*params.w2[i]*sqr(q[i]-params.q_0[i]); }
	return -E;
}

double rand_state(double *q, size_t dim, gsl_rng *r, TSHO &params) {
	for(size_t i=0; i<dim; i++) { q[i] = gsl_ran_gaussian_ziggurat(r, 10.); }
}

int main(int argc, char **argv) {
	TSHO sho;
	
	sho.add_oscillator(0., 1.);
	sho.add_oscillator(1., 2.);
	sho.add_oscillator(-1., 3.);
	
	double logger;
	
	THybridMC<TSHO, double> hmc(sho.dim, negE_SHO, rand_state, sho, logger);
	
	hmc.leapfrog(100, 0.01);
	
	return 0;
}

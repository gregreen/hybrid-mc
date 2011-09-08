#ifndef __HYBRIDMC_H_
#define __HYBRIDMC_H_


#include <iostream>
#include <math.h>

#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>


void seed_gsl_rng(gsl_rng **r);	// Seed a gsl_rng with the Unix time in nanoseconds


template<class TParams, class TLogger>
class THybridMC {
	double *q, *p;		// Position and momentum
	size_t dim;		// Dimensionality of parameter space
	TParams &params;	// Additional parameters passed to pdf
	TLogger &logger;	// Object which gets passed elements in chain
	
	gsl_rng *r;		// Random number generator
	
public:
	typedef double (*log_pdf_t)(const double *const q, size_t dim, TParams &params);
	typedef double (*rand_state_t)(double *q, size_t dim, gsl_rng *r, TParams &params);
	
	THybridMC(size_t _dim, log_pdf_t _log_pdf, rand_state_t _rand_state, TParams &_params, TLogger &_logger);
	~THybridMC();
	
	void leapfrog(unsigned int N_steps, double eta);
	
private:
	log_pdf_t log_pdf;
	rand_state_t rand_state;
	
	void draw_p();
	//void leapfrog(unsigned int N_steps, double eta);
	double delE_delqi(double eta, size_t i);
};


template<class TParams, class TLogger>
THybridMC<TParams, TLogger>::THybridMC(size_t _dim, log_pdf_t _log_pdf, rand_state_t _rand_state, TParams &_params, TLogger &_logger)
	: dim(_dim), log_pdf(_log_pdf), rand_state(_rand_state), params(_params), logger(_logger), q(NULL), p(NULL), r(NULL)
{
	seed_gsl_rng(&r);
	
	q = new double[dim];
	p = new double[dim];
	
	rand_state(q, dim, r, params);
	draw_p();
}

template<class TParams, class TLogger>
THybridMC<TParams, TLogger>::~THybridMC() {
	delete [] q;
	delete [] p;
	gsl_rng_free(r);
}

template<class TParams, class TLogger>
inline void THybridMC<TParams, TLogger>::draw_p() {
	for(size_t i=0; i<dim; i++) { p[i] = gsl_ran_gaussian_ziggurat(r, 1.); }
}

template<class TParams, class TLogger>
inline void THybridMC<TParams, TLogger>::leapfrog(unsigned int N_steps, double eta) {
	// Half-step in p
	for(size_t i=0; i<dim; i++) { p[i] -= 0.5*eta*delE_delqi(eta, i); }
	for(unsigned int n=0; n<N_steps; n++) {
		// Full step in q
		for(size_t i=0; i<dim; i++) { q[i] += eta*p[i]; }
		// Full step in p
		for(size_t i=0; i<dim; i++) { p[i] -= eta*delE_delqi(eta, i); }
	}
}

// TODO: Deal with boundaries where E -> +infinity
// Estimate delE/delq_i by sampling E at q_i +- eta/10
template<class TParams, class TLogger>
inline double THybridMC<TParams, TLogger>::delE_delqi(double eta, size_t i) {
	double q_i_tmp = q[i];
	q[i] += eta/10.;
	double delta_E = log_pdf(q, dim, params);
	q[i] = q_i_tmp - eta/10.;
	delta_E -= log_pdf(q, dim, params);
	q[i] = q_i_tmp;
	return 5.*delta_E/eta;
}



// Seed a gsl_rng with the Unix time in nanoseconds
inline void seed_gsl_rng(gsl_rng **r) {
	timespec t_seed;
	clock_gettime(CLOCK_REALTIME, &t_seed);
	long unsigned int seed = 1e9*(long unsigned int)t_seed.tv_sec;
	seed += t_seed.tv_nsec;
	*r = gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set(*r, seed);
}

#endif // __HYBRIDMC_H_
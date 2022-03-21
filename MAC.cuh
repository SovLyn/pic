#pragma once
#ifndef MAC_CUH
#define MAC_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include "cudaMemoryManager.cuh"
#include <boost/core/noncopyable.hpp>
#include "cudaErrorCheck.cuh"

class MACpoints:private boost::noncopyable{
public:
	MACpoints(unsigned int Nx, unsigned int Ny, unsigned int Nz);
	virtual ~MACpoints();
	cudaR::cudaUM<double> u, uold;
	cudaR::cudaUM<unsigned int> m;
private:
	const unsigned int Nx, Ny, Nz;
};

/*! \enum gridType
 *
 *  Descripts the type of each block
 */
enum gridType { Empty, Fuild, Solid };

class MAC:private boost::noncopyable{
public:
	MAC(unsigned int Nx, unsigned int Ny, unsigned int Nz);
	virtual ~MAC();


	//Have scale Nx*Ny*Nz
	cudaR::cudaUM<float> pressure, divu;
	cudaR::cudaUM<gridType> blockType;
	//Have scale Nx*(Ny+1)*(Nz+1)
	MACpoints x;
	//Have scale (Nx+1)*Ny*(Nz+1)
	MACpoints y;
	//Have scale (Nx+1)*(Ny+1)*Nz
	MACpoints z;

private:
	const unsigned int Nx, Ny, Nz;

};

class Particles:private boost::noncopyable{
public:
	explicit Particles(unsigned int n);
	virtual ~Particles();

	//Temperal struct for two floats and one int;
	struct flflin{
		float _down;
		float _up;
		int _n;
		flflin(float down, float up, int n){
			_down = down;
			_up = up;
			_n = n;
		}
	};
	void setParticlesUniform(flflin xlim, flflin ylim, flflin zlim);
	//Have scale n
	cudaR::cudaUM<float4> position;
	cudaR::cudaUM<float3> velocity;
	unsigned int N()const{return _N;}
private:
	const unsigned int _N;
};

#endif /* MAC_CUH */

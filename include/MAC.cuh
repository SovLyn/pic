#pragma once
#ifndef MAC_CUH
#define MAC_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include "cudaMemoryManager.cuh"
#include <boost/core/noncopyable.hpp>
#include "cudaErrorCheck.cuh"

class MACpoints;
class MAC;
class Particles;

class MACpoints:private boost::noncopyable{
public:
	MACpoints(int Nx, int Ny, int Nz);
	virtual ~MACpoints();
	cudaR::cudaM<float> u, uold, m;
	void applyForce(float f);
private:
	const int _Nx, _Ny, _Nz;
};

/*! \enum gridType
 *
 *  Descripts the type of each block
 */
enum gridType { Empty, Fluid, Solid };

class MAC:private boost::noncopyable{
public:
	MAC(int Nx, int Ny, int Nz, float flip=0.95, float rho=1.0);
	virtual ~MAC();


	//Have scale Nx*Ny*Nz
	cudaR::cudaM<float> pressure, divu;
	cudaR::cudaM<gridType> blockType;
	//Have scale Nx*(Ny+1)*(Nz+1)
	MACpoints x;
	//Have scale (Nx+1)*Ny*(Nz+1)
	MACpoints y;
	//Have scale (Nx+1)*(Ny+1)*Nz
	MACpoints z;

	int Nx()const{return _Nx;}
	int Ny()const{return _Ny;}
	int Nz()const{return _Nz;}

	void particlesToGrid(Particles& parts);
	void gridToParticles(Particles& parts);

	void solvePressure(const int itN, const float dt);

	void applyForce(float3 f, float dt);
private:
	const int _Nx, _Ny, _Nz;
	float _flip;
	float _rho;

};

class Particles:private boost::noncopyable{
public:
	explicit Particles(int n);
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
	cudaR::cudaM<float4> position;
	cudaR::cudaM<float3> velocity;
	int N()const{return _N;}

	//apply const force for particles.
	void applyForce(float3 f, float dt);

	//change the position by velocity.
	void settle(float dt, float Mx, float My, float Mz);
private:
	const int _N;
};

#endif /* MAC_CUH */

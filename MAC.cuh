#pragma once
#ifndef MAC_H
#define MAC_H

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

MACpoints::MACpoints(unsigned int Nx, unsigned int Ny, unsigned int Nz):
Nx(Nx), Ny(Ny), Nz(Nz),
u(Nx*Ny*Nz), uold(Nx*Ny*Nz), m(Nx*Ny*Nz){
	
}

MACpoints::~MACpoints(){}

class MAC:private boost::noncopyable{
public:
	MAC(unsigned int Nx, unsigned int Ny, unsigned int Nz);
	virtual ~MAC();
	//Have scale Nx*Ny*Nz
	cudaR::cudaUM<double> pressure, divu;
	//Have scale Nx*(Ny+1)*(Nz+1)
	MACpoints x;
	//Have scale (Nx+1)*Ny*(Nz+1)
	MACpoints y;
	//Have scale (Nx+1)*(Ny+1)*Nz
	MACpoints z;

private:
	const unsigned int Nx, Ny, Nz;

};

MAC::MAC(unsigned int Nx, unsigned int Ny, unsigned int Nz):
Nx(Nx),Ny(Ny),Nz(Nz),
pressure(Nx*Ny*Nz), divu(Nx*Ny*Nz),
x(Nx, Ny+1, Nz+1), y(Nx+1, Ny, Nz+1), z(Nx+1, Ny+1, Nz){

}

MAC::~MAC(){}

class Particles:private boost::noncopyable{
public:
	explicit Particles(unsigned int n);
	virtual ~Particles();
public:
	//Have scale n
	cudaR::cudaUM<float4> position;
	cudaR::cudaUM<float3> velocity;
};

Particles::Particles(unsigned int n):
position(n), velocity(n){

}

Particles::~Particles(){}

#endif /* MAC_H */

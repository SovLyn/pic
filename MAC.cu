#include "MAC.cuh"

MACpoints::MACpoints(unsigned int Nx, unsigned int Ny, unsigned int Nz):
Nx(Nx), Ny(Ny), Nz(Nz),
u(Nx*Ny*Nz), uold(Nx*Ny*Nz), m(Nx*Ny*Nz){
	
}

MACpoints::~MACpoints(){}

MAC::MAC(unsigned int Nx, unsigned int Ny, unsigned int Nz):
Nx(Nx),Ny(Ny),Nz(Nz),
pressure(Nx*Ny*Nz), divu(Nx*Ny*Nz), blockType(Nx*Ny*Nz),
x(Nx, Ny+1, Nz+1), y(Nx+1, Ny, Nz+1), z(Nx+1, Ny+1, Nz){

}

MAC::~MAC(){}

Particles::Particles(unsigned int n):
position(n), velocity(n), _N(n){

}

Particles::~Particles(){}

void Particles::setParticlesUniform(flflin xlim, flflin ylim, flflin zlim){
	float deltax = xlim._n > 1?(xlim._up - xlim._down)/(xlim._n - 1):(xlim._up-xlim._down);
	float deltay = ylim._n > 1?(ylim._up - ylim._down)/(ylim._n - 1):(ylim._up-ylim._down);
	float deltaz = zlim._n > 1?(zlim._up - zlim._down)/(zlim._n - 1):(zlim._up-zlim._down);
	for (int i = 0; i < _N; ++i) {
		int iz = i/(xlim._n*ylim._n);
		int iy = (i-iz*(xlim._n*ylim._n))/xlim._n;
		int ix = (i-iz*(xlim._n*ylim._n)-iy*xlim._n);
		position.p()[i] = float4{
			ix*deltax+xlim._down,
			iy*deltay+ylim._down,
			iz*deltaz+zlim._down,
			1.0
		};
	}
}


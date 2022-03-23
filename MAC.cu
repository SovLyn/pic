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

/*The kernal function for Particles::applyForce.*/
static __global__ void particles_apply_force(float3 *v, float3 df, unsigned int N){
	const unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < N) {
		v[tid].x+=df.x;
		v[tid].y+=df.y;
		v[tid].z+=df.z;
	}

}

void Particles::applyForce(float3 f, float dt){
	const int blockSize = 32;
	f.x*=dt;
	f.y*=dt;
	f.z*=dt;
	particles_apply_force<<<(_N-1)/blockSize+1, blockSize>>>(velocity.p(), f, _N);
}

static __global__ void particles_settle_kernal(float4 *p, float3 *v, float dt, unsigned int N){
	const unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid >= N) {
		return;
	}

	float px = p[tid].x;
	float py = p[tid].y;
	float pz = p[tid].z;

	float vx = v[tid].x;
	float vy = v[tid].y;
	float vz = v[tid].z;

	px+=vx*dt;
	if (px > 1.0) {
		px = 1.0;
		v[tid].x = 0;
	}else if (px < 0.0) {
		px = 0.0;
		v[tid].x = 0;
	}

	py+=vy*dt;
	if (py > 1.0) {
		py = 1.0;
		v[tid].y = -vy;
	}else if (py < 0.0) {
		py = 0.0;
		v[tid].y = 0;
	}

	pz+=vz*dt;
	if (pz > 1.0) {
		pz = 1.0;
		v[tid].z = 0;
	}else if (pz < 0.0) {
		pz = 0.0;
		v[tid].z = 0;
	}

	p[tid].x = px;
	p[tid].y = py;
	p[tid].z = pz;
}

void Particles::settle(float dt){
	const int blockSize = 32;
	particles_settle_kernal<<<(_N-1)/blockSize+1, blockSize>>>(position.p(), velocity.p(), dt, _N);
}

#include "MAC.cuh"

MACpoints::MACpoints(unsigned int Nx, unsigned int Ny, unsigned int Nz):
Nx(Nx), Ny(Ny), Nz(Nz),
u(Nx*Ny*Nz), uold(Nx*Ny*Nz), m(Nx*Ny*Nz){
	
}

MACpoints::~MACpoints(){}

MAC::MAC(unsigned int Nx, unsigned int Ny, unsigned int Nz, float flip):
_Nx(Nx), _Ny(Ny), _Nz(Nz), _flip(flip),
pressure(Nx*Ny*Nz), divu(Nx*Ny*Nz), blockType(Nx*Ny*Nz),
x(Nx+1, Ny, Nz), y(Nx, Ny+1, Nz), z(Nx, Ny, Nz+1){

}

MAC::~MAC(){}

static __global__ void quick_divide(float *a, float *b, unsigned int N){
	const unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid<N) {
		a[tid]/=b[tid];
	}
}

static __global__ void particlesToGridx(float3 *v,
										float4 *p,
										float *ux,
										float *m,
										const unsigned int pN,
										const unsigned int Nx,
										const unsigned int Ny,
										const unsigned int Nz){
	
	const unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid>=pN) {
		return;
	}
	float vx = v[tid].x;

	//Here to change.
	int3 corner = make_int3(
			int(p[tid].x*Nx),
			int(p[tid].y*Ny+0.5)-1,
			int(p[tid].z*Nz+0.5)-1
			);
	for (int i = corner.x; i < corner.x+2; ++i) {
		for (int j = corner.y; j < corner.y+2; ++j) {
			for (int k = corner.z; k < corner.z+2; ++k) {
				if(i>=0&&i<Nx+1&&j>=0&&j<Ny&&k>=0&&k<Nz){
					//Here to change
					float pgx = p[tid].x*Nx;
					float pgy = p[tid].y*Ny-0.5;
					float pgz = p[tid].z*Nz-0.5;
					float scale = (1.0-fabsf(pgx-i))*(1.0-fabsf(pgy-j))*(1.0-fabsf(pgz-k));
//					printf("tid:%d, i:%d, j:%d, k:%d, px:%f, py:%f, pz:%f, fpgx: %f, fpgy: %f, fpgz: %f, scale:%f\n", tid, i, j, k, p[tid].x, p[tid].y, p[tid].z, fabsf(pgx-i), fabsf(pgy-j), fabsf(pgz-k), scale);
					//Here to change.
					atomicAdd(m+k*((Nx+1)*Ny)+j*(Nx+1)+i, scale);
					atomicAdd(ux+k*((Nx+1)*Ny)+j*(Nx+1)+i, scale*vx);
				}
			}
		}
	}

}

static __global__ void particlesToGridy(float3 *v,
										float4 *p,
										float *uy,
										float *m,
										const unsigned int pN,
										const unsigned int Nx,
										const unsigned int Ny,
										const unsigned int Nz){
	
	const unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid>=pN) {
		return;
	}
	//Here to change.
	float vy = v[tid].y;

	//Here to change.
	int3 corner = make_int3(
			int(p[tid].x*Nx+0.5)-1,
			int(p[tid].y*Ny),
			int(p[tid].z*Nz+0.5)-1
			);
	for (int i = corner.x; i < corner.x+2; ++i) {
		for (int j = corner.y; j < corner.y+2; ++j) {
			for (int k = corner.z; k < corner.z+2; ++k) {
				if(i>=0&&i<Nx+1&&j>=0&&j<Ny&&k>=0&&k<Nz){
					//Here to change
					float pgx = p[tid].x*Nx-0.5;
					float pgy = p[tid].y*Ny;
					float pgz = p[tid].z*Nz-0.5;
					float scale = (1.0-fabsf(pgx-i))*(1.0-fabsf(pgy-j))*(1.0-fabsf(pgz-k));
//					printf("tid:%d, i:%d, j:%d, k:%d, px:%f, py:%f, pz:%f, fpgx: %f, fpgy: %f, fpgz: %f, scale:%f\n", tid, i, j, k, p[tid].x, p[tid].y, p[tid].z, fabsf(pgx-i), fabsf(pgy-j), fabsf(pgz-k), scale);
					//Here to change.
					atomicAdd(m+k*((Nx)*(Ny+1))+j*(Nx)+i, scale);
					atomicAdd(uy+k*((Nx)*(Ny+1))+j*(Nx)+i, scale*vy);
				}
			}
		}
	}

}

static __global__ void particlesToGridz(float3 *v,
										float4 *p,
										float *uz,
										float *m,
										const unsigned int pN,
										const unsigned int Nx,
										const unsigned int Ny,
										const unsigned int Nz){
	
	const unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid>=pN) {
		return;
	}
	//Here to change.
	float vz = v[tid].z;

	//Here to change.
	int3 corner = make_int3(
			int(p[tid].x*Nx+0.5)-1,
			int(p[tid].y*Ny+0.5)-1,
			int(p[tid].z*Nz)
			);
	for (int i = corner.x; i < corner.x+2; ++i) {
		for (int j = corner.y; j < corner.y+2; ++j) {
			for (int k = corner.z; k < corner.z+2; ++k) {
				if(i>=0&&i<Nx+1&&j>=0&&j<Ny&&k>=0&&k<Nz){
					//Here to change
					float pgx = p[tid].x*Nx;
					float pgy = p[tid].y*Ny;
					float pgz = p[tid].z*Nz-0.5;
					float scale = (1.0-fabsf(pgx-i))*(1.0-fabsf(pgy-j))*(1.0-fabsf(pgz-k));
//					printf("tid:%d, i:%d, j:%d, k:%d, px:%f, py:%f, pz:%f, fpgx: %f, fpgy: %f, fpgz: %f, scale:%f\n", tid, i, j, k, p[tid].x, p[tid].y, p[tid].z, fabsf(pgx-i), fabsf(pgy-j), fabsf(pgz-k), scale);
					//Here to change.
					atomicAdd(m+k*((Nx)*(Ny))+j*(Nx)+i, scale);
					atomicAdd(uz+k*((Nx)*(Ny))+j*(Nx)+i, scale*vz);
				}
			}
		}
	}

}

static __global__ void set_grid_type(gridType *gT, float4 *p, unsigned int pN, unsigned int Nx, unsigned int Ny, unsigned int Nz){
	const unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid<pN) {
		unsigned int i = int(p[tid].x*Nx);
		unsigned int j = int(p[tid].y*Ny);
		unsigned int k = int(p[tid].z*Nz);
		if (gT[k*Ny*Nx+j*Nx*i] == gridType::Empty) {
			gT[k*Ny*Nx+j*Nx*i] = gridType::Fuild;
		}
	}
}

static __global__ void clean_block_type(gridType *gT, const unsigned int N){
	const unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < N) {
		if (gT[tid]==gridType::Fuild) {
			gT[tid]=gridType::Empty;
		}
	}
}

void MAC::particlesToGrid(Particles& parts){
	const int blockSize = 32;

	clean_block_type<<<(_Nx*_Ny*_Nz-1)/128+1, 128>>>(this->blockType.p(), _Nx*_Ny*_Nz);
	set_grid_type<<<(_Nx*_Ny*_Nz-1)/128+1, 128>>>(
			this->blockType.p(),
			parts.position.p(),
			parts.N(), _Nx, _Ny, _Nz);

	this->x.u.setVal(0.0);
	this->x.m.setVal(0.0);
	particlesToGridx<<<(parts.N()-1)/blockSize+1, blockSize>>>(
			parts.velocity.p(), parts.position.p(),
			this->x.u.p(), this->x.m.p(),
			parts.N(), _Nx, _Ny, _Nz);
	quick_divide<<<((_Nx+1)*_Ny*_Nz-1)/128+1, 128>>>(this->x.u.p(), this->x.m.p(), (_Nx+1)*_Ny*_Nz);
	CHECK(cudaMemcpy(this->x.uold.p(), this->x.u.p(), sizeof(float)*(_Nx+1)*_Ny*_Nz, cudaMemcpyDeviceToDevice));

	this->y.u.setVal(0.0);
	this->y.m.setVal(0.0);
	particlesToGridy<<<(parts.N()-1)/blockSize+1, blockSize>>>(
			parts.velocity.p(), parts.position.p(),
			this->y.u.p(), this->y.m.p(),
			parts.N(), _Nx, _Ny, _Nz);
	quick_divide<<<((_Nx)*(_Ny+1)*_Nz-1)/128+1, 128>>>(this->y.u.p(), this->y.m.p(), (_Nx)*(_Ny+1)*_Nz);
	CHECK(cudaMemcpy(this->y.uold.p(), this->y.u.p(), sizeof(float)*(_Nx)*(_Ny+1)*_Nz, cudaMemcpyDeviceToDevice));

	this->z.u.setVal(0.0);
	this->z.m.setVal(0.0);
	particlesToGridz<<<(parts.N()-1)/blockSize+1, blockSize>>>(
			parts.velocity.p(), parts.position.p(),
			this->z.u.p(), this->z.m.p(),
			parts.N(), _Nx, _Ny, _Nz);
	quick_divide<<<((_Nx)*_Ny*(_Nz+1)-1)/128+1, 128>>>(this->z.u.p(), this->z.m.p(), (_Nx)*_Ny*(_Nz+1));
	CHECK(cudaMemcpy(this->z.uold.p(), this->z.u.p(), sizeof(float)*(_Nx)*_Ny*(_Nz+1), cudaMemcpyDeviceToDevice))
}

static __global__ void gridToParticlesx(float3 *v,
										float4 *p,
										float *ux,
										float *uxold,
										const unsigned int pN,
										const unsigned int Nx,
										const unsigned int Ny,
										const unsigned int Nz,
										const float flip){
	
	const unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid>=pN) {
		return;
	}
	//Here to change.
	float vx = 0.0;
	float vxold = 0.0;

	//Here to change.
	int3 corner = make_int3(
			int(p[tid].x*Nx),
			int(p[tid].y*Ny+0.5)-1,
			int(p[tid].z*Nz+0.5)-1
			);

	for (int i = corner.x; i < corner.x+2; ++i) {
		for (int j = corner.y; j < corner.y+2; ++j) {
			for (int k = corner.z; k < corner.z+2; ++k) {
				if(i>=0&&i<Nx+1&&j>=0&&j<Ny&&k>=0&&k<Nz){
					//Here to change
					float pgx = p[tid].x*Nx;
					float pgy = p[tid].y*Ny-0.5;
					float pgz = p[tid].z*Nz-0.5;
					float scale = (1.0-fabsf(pgx-i))*(1.0-fabsf(pgy-j))*(1.0-fabsf(pgz-k));
					//Here to change.
					vx += scale*ux[k*((Nx+1)*Ny)+j*(Nx+1)+i];
					vxold += scale*uxold[k*((Nx+1)*Ny)+j*(Nx+1)+i];
				}
			}
		}
	}

	//Here to change.
	//v[tid].x = (1-flip)*vx + flip*(v[tid]+vx-vxold);
	v[tid].x = vx + flip*(v[tid].x-vxold);

}

static __global__ void gridToParticlesy(float3 *v,
										float4 *p,
										float *uy,
										float *uyold,
										const unsigned int pN,
										const unsigned int Nx,
										const unsigned int Ny,
										const unsigned int Nz,
										const float flip){
	
	const unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid>=pN) {
		return;
	}
	//Here to change.
	float vy = 0.0;
	float vyold = 0.0;

	//Here to change.
	int3 corner = make_int3(
			int(p[tid].x*Nx+0.5)-1,
			int(p[tid].y*Ny),
			int(p[tid].z*Nz+0.5)-1
			);

	for (int i = corner.x; i < corner.x+2; ++i) {
		for (int j = corner.y; j < corner.y+2; ++j) {
			for (int k = corner.z; k < corner.z+2; ++k) {
				if(i>=0&&i<Nx+1&&j>=0&&j<Ny&&k>=0&&k<Nz){
					//Here to change
					float pgx = p[tid].x*Nx-0.5;
					float pgy = p[tid].y*Ny;
					float pgz = p[tid].z*Nz-0.5;
					float scale = (1.0-fabsf(pgx-i))*(1.0-fabsf(pgy-j))*(1.0-fabsf(pgz-k));
					//Here to change.
					vy += scale*uy[k*((Nx)*(Ny+1))+j*(Nx)+i];
					vyold += scale*uyold[k*((Nx)*(Ny+1))+j*(Nx)+i];
				}
			}
		}
	}

	//Here to change.
	//v[tid].y = (1-flip)*vy + flip*(v[tid]+vy-vyold);
	v[tid].y = vy + flip*(v[tid].y-vyold);

}

static __global__ void gridToParticlesz(float3 *v,
										float4 *p,
										float *uz,
										float *uzold,
										const unsigned int pN,
										const unsigned int Nx,
										const unsigned int Ny,
										const unsigned int Nz,
										const float flip){
	
	const unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid>=pN) {
		return;
	}
	//Here to change.
	float vz = 0.0;
	float vzold = 0.0;

	//Here to change.
	int3 corner = make_int3(
			int(p[tid].x*Nx+0.5)-1,
			int(p[tid].y*Ny+0.5)-1,
			int(p[tid].z*Nz)
			);

	for (int i = corner.x; i < corner.x+2; ++i) {
		for (int j = corner.y; j < corner.y+2; ++j) {
			for (int k = corner.z; k < corner.z+2; ++k) {
				if(i>=0&&i<Nx+1&&j>=0&&j<Ny&&k>=0&&k<Nz){
					//Here to change
					float pgx = p[tid].x*Nx;
					float pgy = p[tid].y*Ny-0.5;
					float pgz = p[tid].z*Nz-0.5;
					float scale = (1.0-fabsf(pgx-i))*(1.0-fabsf(pgy-j))*(1.0-fabsf(pgz-k));
					//Here to change.
					vz += scale*uz[k*((Nx)*Ny)+j*(Nx)+i];
					vzold += scale*uzold[k*((Nx)*Ny)+j*(Nx)+i];
				}
			}
		}
	}

	//Here to change.
	//v[tid].x = (1-flip)*vx + flip*(v[tid]+vx-vxold);
	v[tid].z = vz + flip*(v[tid].z-vzold);

}

void MAC::gridToParticles(Particles& parts){
	const int blockSize = 32;
	gridToParticlesx<<<(parts.N()-1)/blockSize+1, blockSize>>>(
			parts.velocity.p(), parts.position.p(),
			this->x.u.p(), this->x.uold.p(),
			parts.N(), _Nx, _Ny, _Nz, _flip);
	gridToParticlesy<<<(parts.N()-1)/blockSize+1, blockSize>>>(
			parts.velocity.p(), parts.position.p(),
			this->y.u.p(), this->y.uold.p(),
			parts.N(), _Nx, _Ny, _Nz, _flip);
	gridToParticlesz<<<(parts.N()-1)/blockSize+1, blockSize>>>(
			parts.velocity.p(), parts.position.p(),
			this->z.u.p(), this->z.uold.p(),
			parts.N(), _Nx, _Ny, _Nz, _flip);
}

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
static __global__ void particles_apply_force(float3 *v, float3 df, const unsigned int N){
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

static __global__ void particles_settle_kernal(float4 *p, float3 *v, float dt, const unsigned int N){
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

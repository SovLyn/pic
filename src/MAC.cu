#include "MAC.cuh"

MACpoints::MACpoints(int Nx, int Ny, int Nz):
_Nx(Nx), _Ny(Ny), _Nz(Nz),
u(Nx*Ny*Nz), uold(Nx*Ny*Nz), m(Nx*Ny*Nz){
	
}

MACpoints::~MACpoints(){}

static __global__ void apply_force(float *u, float *m, float f, const int N){
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid<N&&m[tid]>1e-9) {
		u[tid]=u[tid]+f;
	}
}

void MACpoints::applyForce(float f){
	const int blockSize = 128;
	apply_force<<<(_Nx*_Ny*_Nz-1)/blockSize+1, blockSize>>>(this->u.p(), this->m.p(), f, _Nx*_Ny*_Nz);
}

static __global__ void set_solid(gridType *gT, const int Nx, const int Ny, const int Nz){
	const int i = blockDim.x*blockIdx.x + threadIdx.x;
	const int j = blockDim.y*blockIdx.y + threadIdx.y;
	const int k = blockDim.z*blockIdx.z + threadIdx.z;
	if(i>(Nx-1)||j>(Ny-1)||k>(Nz-1))return;
	gT[k*Nx*Ny+j*Nx+i] = (i==0||j==0||k==0||i==(Nx-1)||j==(Ny-1)||k==(Nz-1))?(gridType::Solid) : (gridType::Empty);

}

MAC::MAC(int Nx, int Ny, int Nz, float flip, float rho):
_Nx(Nx), _Ny(Ny), _Nz(Nz), _flip(flip), _rho(rho),
pressure(Nx*Ny*Nz), divu(Nx*Ny*Nz), blockType(Nx*Ny*Nz),
x(Nx+1, Ny, Nz), y(Nx, Ny+1, Nz), z(Nx, Ny, Nz+1){
	int blockSize = 4;
	set_solid<<<
		dim3((_Nx-1)/blockSize+1, (_Ny-1)/blockSize+1, (_Nz-1)/blockSize), dim3(blockSize, blockSize, blockSize)>>>(
				blockType.p(), _Nx, _Ny, _Nz);
}

MAC::~MAC(){}

void MAC::applyForce(float3 f, float dt){
	this->x.applyForce(f.x*dt);
	this->y.applyForce(f.y*dt);
	this->z.applyForce(f.z*dt);
}

static __global__ void quick_divide(float *a, float *b, int N){
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid<N&&b[tid]>1e-9) {
		a[tid]=a[tid]/b[tid];
	}
}

static __global__ void particlesToGridx(float3 *v,
										float4 *p,
										float *ux,
										float *m,
										const int pN,
										const int Nx,
										const int Ny,
										const int Nz){
	
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid>=pN) {
		return;
	}
	float vx = v[tid].x;
	float4 pos = p[tid];

	//Here to change.
	int3 corner = make_int3(
			int(pos.x),
			int(pos.y+0.5)-1,
			int(pos.z+0.5)-1
			);
	for (int i = corner.x; i < corner.x+2; ++i) {
		for (int j = corner.y; j < corner.y+2; ++j) {
			for (int k = corner.z; k < corner.z+2; ++k) {
				//Here to change
				float pgx = pos.x;
				float pgy = pos.y-0.5;
				float pgz = pos.z-0.5;
				float scale = (1.0-fabsf(pgx-i))*(1.0-fabsf(pgy-j))*(1.0-fabsf(pgz-k));
				//Here to change.
				atomicAdd(m+k*((Nx+1)*Ny)+j*(Nx+1)+i, scale);
				atomicAdd(ux+k*((Nx+1)*Ny)+j*(Nx+1)+i, scale*vx);
			}
		}
	}

}

static __global__ void particlesToGridy(float3 *v,
										float4 *p,
										float *uy,
										float *m,
										const int pN,
										const int Nx,
										const int Ny,
										const int Nz){
	
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid>=pN) {
		return;
	}
	//Here to change.
	float vy = v[tid].y;
	float4 pos = p[tid];

	//Here to change.
	int3 corner = make_int3(
			int(pos.x+0.5)-1,
			int(pos.y),
			int(pos.z+0.5)-1
			);
	for (int i = corner.x; i < corner.x+2; ++i) {
		for (int j = corner.y; j < corner.y+2; ++j) {
			for (int k = corner.z; k < corner.z+2; ++k) {
				//Here to change
				float pgx = pos.x-0.5;
				float pgy = pos.y;
				float pgz = pos.z-0.5;
				float scale = (1.0-fabsf(pgx-i))*(1.0-fabsf(pgy-j))*(1.0-fabsf(pgz-k));
				//Here to change.
				atomicAdd(m+k*((Nx)*(Ny+1))+j*(Nx)+i, scale);
				atomicAdd(uy+k*((Nx)*(Ny+1))+j*(Nx)+i, scale*vy);
			}
		}
	}

}

static __global__ void particlesToGridz(float3 *v,
										float4 *p,
										float *uz,
										float *m,
										const int pN,
										const int Nx,
										const int Ny,
										const int Nz){
	
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid>=pN) {
		return;
	}
	//Here to change.
	float vz = v[tid].z;
	float4 pos = p[tid];

	//Here to change.
	int3 corner = make_int3(
			int(pos.x+0.5)-1,
			int(pos.y+0.5)-1,
			int(pos.z)
			);
	for (int i = corner.x; i < corner.x+2; ++i) {
		for (int j = corner.y; j < corner.y+2; ++j) {
			for (int k = corner.z; k < corner.z+2; ++k) {
				//Here to change
				float pgx = pos.x-0.5;
				float pgy = pos.y-0.5;
				float pgz = pos.z;
				float scale = (1.0-fabsf(pgx-i))*(1.0-fabsf(pgy-j))*(1.0-fabsf(pgz-k));
				//Here to change.
				atomicAdd(m+k*((Nx)*(Ny))+j*(Nx)+i, scale);
				atomicAdd(uz+k*((Nx)*(Ny))+j*(Nx)+i, scale*vz);
			}
		}
	}

}

static __global__ void set_grid_type(gridType *gT, float4 *p, int pN, int Nx, int Ny, int Nz){
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid<pN) {
		int i = int(p[tid].x);
		int j = int(p[tid].y);
		int k = int(p[tid].z);
		const int index = k*Nx*Ny+j*Nx+i;
		if (gT[index] != gridType::Solid) {
			gT[index] = gridType::Fluid;
		}
	}
}

static __global__ void clean_block_type(gridType *gT, const int N){
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < N) {
		if (gT[tid]==gridType::Fluid) {
			gT[tid]=gridType::Empty;
		}
	}
}

void MAC::particlesToGrid(Particles& parts){
	const int blockSize = 32;

	clean_block_type<<<(_Nx*_Ny*_Nz-1)/128+1, 128>>>(this->blockType.p(), _Nx*_Ny*_Nz);
	set_grid_type<<<(parts.N()-1)/128+1, 128>>>(
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
										const int pN,
										const int Nx,
										const int Ny,
										const int Nz,
										const float flip){
	
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid>=pN) {
		return;
	}
	//Here to change.
	float vx = 0.0;
	float vxold = 0.0;

	//Here to change.
	int3 corner = make_int3(
			int(p[tid].x),
			int(p[tid].y+0.5)-1,
			int(p[tid].z+0.5)-1
			);

	for (int i = corner.x; i < corner.x+2; ++i) {
		for (int j = corner.y; j < corner.y+2; ++j) {
			for (int k = corner.z; k < corner.z+2; ++k) {
				if(i>=0&&i<(Nx+1)&&j>=0&&j<Ny&&k>=0&&k<Nz){
					//Here to change
					float pgx = p[tid].x;
					float pgy = p[tid].y-0.5;
					float pgz = p[tid].z-0.5;
					float scale = (1.0-fabsf(pgx-i))*(1.0-fabsf(pgy-j))*(1.0-fabsf(pgz-k));
					//Here to change.
					vx += scale*ux[k*((Nx+1)*Ny)+j*(Nx+1)+i];
					vxold += scale*uxold[k*((Nx+1)*Ny)+j*(Nx+1)+i];
				}
			}
		}
	}

	//Here to change.
//	v[tid].x = (1-flip)*vx + flip*(v[tid].x+vx-vxold);
	v[tid].x = vx + flip*(v[tid].x-vxold);

}

static __global__ void gridToParticlesy(float3 *v,
										float4 *p,
										float *uy,
										float *uyold,
										const int pN,
										const int Nx,
										const int Ny,
										const int Nz,
										const float flip){
	
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid>=pN) {
		return;
	}
	//Here to change.
	float vy = 0.0;
	float vyold = 0.0;

	//Here to change.
	int3 corner = make_int3(
			int(p[tid].x+0.5)-1,
			int(p[tid].y),
			int(p[tid].z+0.5)-1
			);

	for (int i = corner.x; i < corner.x+2; ++i) {
		for (int j = corner.y; j < corner.y+2; ++j) {
			for (int k = corner.z; k < corner.z+2; ++k) {
				if(i>=0&&i<Nx&&j>=0&&j<(Ny+1)&&k>=0&&k<Nz){
					//Here to change
					float pgx = p[tid].x-0.5;
					float pgy = p[tid].y;
					float pgz = p[tid].z-0.5;
					float scale = (1.0-fabsf(pgx-i))*(1.0-fabsf(pgy-j))*(1.0-fabsf(pgz-k));
					//Here to change.
					vy += scale*uy[k*((Nx)*(Ny+1))+j*(Nx)+i];
					vyold += scale*uyold[k*((Nx)*(Ny+1))+j*(Nx)+i];
				}
			}
		}
	}

	//Here to change.
//	v[tid].y = (1-flip)*vy + flip*(v[tid].y+vy-vyold);
	v[tid].y = vy + flip*(v[tid].y-vyold);

}

static __global__ void gridToParticlesz(float3 *v,
										float4 *p,
										float *uz,
										float *uzold,
										const int pN,
										const int Nx,
										const int Ny,
										const int Nz,
										const float flip){
	
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid>=pN) {
		return;
	}
	//Here to change.
	float vz = 0.0;
	float vzold = 0.0;

	//Here to change.
	int3 corner = make_int3(
			int(p[tid].x+0.5)-1,
			int(p[tid].y+0.5)-1,
			int(p[tid].z)
			);

	for (int i = corner.x; i < corner.x+2; ++i) {
		for (int j = corner.y; j < corner.y+2; ++j) {
			for (int k = corner.z; k < corner.z+2; ++k) {
				if(i>=0&&i<Nx&&j>=0&&j<Ny&&k>=0&&k<(Nz+1)){
					//Here to change
					float pgx = p[tid].x-0.5;
					float pgy = p[tid].y-0.5;
					float pgz = p[tid].z;
					float scale = (1.0-fabsf(pgx-i))*(1.0-fabsf(pgy-j))*(1.0-fabsf(pgz-k));
					//Here to change.
					vz += scale*uz[k*((Nx)*Ny)+j*(Nx)+i];
					vzold += scale*uzold[k*((Nx)*Ny)+j*(Nx)+i];
//					if(k==1)printf("i:%d, j:%d, k:%d, uz:%f, uzold:%f, pgx:%f, pgy:%f, pgz:%f, scale:%f\n", i, j, k, uz[k*Nx*Ny+j*Nx+i], uzold[k*Nx*Ny+j*Nx*i], pgx, pgy, pgz, scale);
				}
			}
		}
	}

	//Here to change.
//	v[tid].z = (1-flip)*vz + flip*(v[tid].z+vz-vzold);
	v[tid].z = vz + flip*(v[tid].z-vzold);
//	printf("tid:%d, cx:%d, cy:%d, cz:%d, vz:%f, vzold:%f\n", tid, corner.x, corner.y, corner.z, vz, vzold);

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

static __global__ void getdivu(float *divu,
							   float *ux, float *uy, float *uz,
							   const int Nx, const int Ny, const int Nz){
	
	const int i = blockDim.x*blockIdx.x + threadIdx.x + 1;
	const int j = blockDim.y*blockIdx.y + threadIdx.y + 1;
	const int k = blockDim.z*blockIdx.z + threadIdx.z + 1;

	if (i>Nx-2||j>Ny-2||k>Nz-2) {
		return;
	}

	float tempdivu = 0.0;
	if(i!=1)tempdivu += ux[k*(Nx+1)*(Ny)+j*(Nx+1)+i];
	if(i!=Nx-2)tempdivu -= ux[k*(Nx+1)*(Ny)+j*(Nx+1)+i+1];
	if(j!=1)tempdivu += uy[k*(Nx)*(Ny+1)+j*(Nx)+i];
	if(j!=Ny-2)tempdivu -= uy[k*(Nx)*(Ny+1)+(j+1)*(Nx)+i];
	if(k!=1)tempdivu += uz[(k)*(Nx)*(Ny)+j*(Nx)+i];
	if(k!=Nz-2)tempdivu -= uz[(k+1)*(Nx)*(Ny)+j*(Nx)+i];

	divu[k*(Nx)*(Ny)+j*(Nx)+i]=tempdivu;
}

enum _fc {isFluid, XS, XP, YS, YP, ZS, ZP, NG};

static __global__ void GSiteration(float* divu, float *p, gridType *gT, const int itN,
								   int Nx, int Ny, int Nz, const float scale){
	const int i = blockDim.x*blockIdx.x + threadIdx.x + 1;
	const int j = blockDim.y*blockIdx.y + threadIdx.y + 1;
	const int k = blockDim.z*blockIdx.z + threadIdx.z + 1;
	unsigned int flag = 0;

	if (i>Nx-2||j>Ny-2||k>Nz-2) {
		flag|=1<<_fc::NG;
	}

	const int index = k*(Nx)*(Ny)+j*(Nx)+i;
	float a=0.0;
	if (!(flag&1<<_fc::NG)) {
		if (gT[index]==gridType::Fluid) {
			flag|=1<<_fc::isFluid;

			if (i!=1/* && gT[k*(Nx)*(Ny)+j*(Nx)+i-1]!=gridType::Solid*/) {
				a+=1.0;
				flag|=1<<_fc::XS;
			}
			if (i!=Nx-2/*&& gT[k*(Nx)*(Ny)+j*(Nx)+i+1]!=gridType::Solid*/) {
				a+=1.0;
				flag|=1<<_fc::XP;
			}
			if (j!=1/*&& gT[k*(Nx)*(Ny)+(j-1)*(Nx)+i]!=gridType::Solid*/) {
				a+=1.0;
				flag|=1<<_fc::YS;
			}
			if (j!=Ny-2/*&& gT[k*(Nx)*(Ny)+(j+1)*(Nx)+i]!=gridType::Solid*/) {
				a+=1.0;
				flag|=1<<_fc::YP;
			}
			if (k!=1/*&& gT[(k-1)*(Nx)*(Ny)+j*(Nx)+i]!=gridType::Solid*/) {
				a+=1.0;
				flag|=1<<_fc::ZS;
			}
			if (k!=Nz-2/* && gT[(k+1)*(Nx)*(Ny)+j*(Nx)+i]!=gridType::Solid*/) {
				a+=1.0;
				flag|=1<<_fc::ZP;
			}
		}else{
			p[index] = 0.0;
		}
		
	}

	for (int n = 0; n < itN; ++n) {
		float nextp = 0.0;
		if(flag&1<<_fc::isFluid){
			nextp = scale*divu[index];
			if(flag&1<<_fc::XS)nextp+=p[index-1];
			if(flag&1<<_fc::XP)nextp+=p[index+1];
			if(flag&1<<_fc::YS)nextp+=p[index-Nx];
			if(flag&1<<_fc::YP)nextp+=p[index+Nx];
			if(flag&1<<_fc::ZS)nextp+=p[index-Nx*Ny];
			if(flag&1<<_fc::ZP)nextp+=p[index+Nx*Ny];
			nextp/=a;
		}

		__syncthreads();

		if (flag&1<<_fc::isFluid) {
			p[index] = nextp;
		}
	}
//	if(/*flag&1<<_fc::isFluid &&*/ k==1)
//		printf("i:%d, j:%d, k:%d, p:%f, divu:%f\n", i, j, k, p[index], divu[index]);
}

static __global__ void get_new_ux(float *ux, float *p,
		const int Nx, const int Ny, const int Nz, const float scale){
	const int i = blockDim.x*blockIdx.x + threadIdx.x + 1;
	const int j = blockDim.y*blockIdx.y + threadIdx.y + 1;
	const int k = blockDim.z*blockIdx.z + threadIdx.z + 1;
	const int index = k*(Nx+1)*(Ny)+j*(Nx+1)+i;

	if (i>(Nx-1)||j>(Ny-2)||k>(Nz-2)) {
		return;
	}

	if (i==1||i==Nx-1) {
		ux[index]=0.0;
	}else{
		ux[index]=ux[index]-scale*(p[k*Nx*Ny+j*Nx+i]-p[k*Nx*Ny+j*Nx+i-1]);
	}
}

static __global__ void get_new_uy(float *uy, float *p,
		const int Nx, const int Ny, const int Nz, const float scale){
	const int i = blockDim.x*blockIdx.x + threadIdx.x + 1;
	const int j = blockDim.y*blockIdx.y + threadIdx.y + 1;
	const int k = blockDim.z*blockIdx.z + threadIdx.z + 1;
	const int index = k*(Nx)*(Ny+1)+j*(Nx)+i;

	if (i>(Nx-2)||j>(Ny-1)||k>(Nz-2)) {
		return;
	}

	if (j==1||j==Ny-1) {
		uy[index]=0.0;
	}else{
		uy[index]=uy[index]-scale*(p[k*Nx*Ny+j*Nx+i]-p[k*Nx*Ny+(j-1)*Nx+i]);
	}
}

static __global__ void get_new_uz(float *uz, float *p,
		const int Nx, const int Ny, const int Nz, const float scale, gridType *gT){
	const int i = blockDim.x*blockIdx.x + threadIdx.x + 1;
	const int j = blockDim.y*blockIdx.y + threadIdx.y + 1;
	const int k = blockDim.z*blockIdx.z + threadIdx.z + 1;
	const int index = k*(Nx)*(Ny)+j*(Nx)+i;

	if (i>(Nx-2)||j>(Ny-2)||k>(Nz-1)) {
		return;
	}

	if (k==1||k==Nz-1) {
		uz[index]=0.0;
	}else{
		uz[index]=uz[index]-scale*(p[k*Nx*Ny+j*Nx+i]-p[(k-1)*Nx*Ny+j*Nx+i]);
	}
}

void MAC::solvePressure(const int itN, const float dt){
	int blockSize = 4;
	getdivu<<<
		dim3((_Nx-3)/blockSize+1, (_Ny-3)/blockSize+1, (_Nz-3)/blockSize), dim3(blockSize, blockSize, blockSize)>>>(
		this->divu.p(), this->x.u.p(), this->y.u.p(), this->z.u.p(), _Nx, _Ny, _Nz);

//	this->pressure.setVal(0.0);
	GSiteration<<<
		dim3((_Nx-3)/blockSize+1, (_Ny-3)/blockSize+1, (_Nz-3)/blockSize+1),
		dim3(blockSize, blockSize, blockSize)>>>(
		this->divu.p(), this->pressure.p(), this->blockType.p(), itN, _Nx, _Ny, _Nz, _rho/dt);

	get_new_ux<<<
		dim3((_Nx-2)/blockSize+1, (_Ny-3)/blockSize+1, (_Nz-3)/blockSize+1),
		dim3(blockSize, blockSize, blockSize)>>>(this->x.u.p(), this->pressure.p(), _Nx, _Ny, _Nz, dt/_rho);
	get_new_uy<<<
		dim3((_Nx-3)/blockSize+1, (_Ny-2)/blockSize+1, (_Nz-3)/blockSize+1),
		dim3(blockSize, blockSize, blockSize)>>>(this->y.u.p(), this->pressure.p(), _Nx, _Ny, _Nz, dt/_rho);
	get_new_uz<<<
		dim3((_Nx-3)/blockSize+1, (_Ny-3)/blockSize+1, (_Nz-2)/blockSize+1),
		dim3(blockSize, blockSize, blockSize)>>>(this->z.u.p(), this->pressure.p(), _Nx, _Ny, _Nz, dt/_rho, this->blockType.p());
}

Particles::Particles(int n):
position(n), velocity(n), _N(n){

}

Particles::~Particles(){}

void Particles::setParticlesUniform(flflin xlim, flflin ylim, flflin zlim){
	float4 *temp_p = new float4[_N];
	float deltax = xlim._n > 1?(xlim._up - xlim._down)/(xlim._n - 1):(xlim._up-xlim._down);
	float deltay = ylim._n > 1?(ylim._up - ylim._down)/(ylim._n - 1):(ylim._up-ylim._down);
	float deltaz = zlim._n > 1?(zlim._up - zlim._down)/(zlim._n - 1):(zlim._up-zlim._down);
	for (int i = 0; i < _N; ++i) {
		int iz = i/(xlim._n*ylim._n);
		int iy = (i-iz*(xlim._n*ylim._n))/xlim._n;
		int ix = (i-iz*(xlim._n*ylim._n)-iy*xlim._n);
		temp_p[i] = float4{
			ix*deltax+xlim._down,
			iy*deltay+ylim._down,
			iz*deltaz+zlim._down,
			1.0
		};
	}
	CHECK(cudaMemcpy(position.p(), temp_p, sizeof(float4)*_N, cudaMemcpyHostToDevice));
	delete[] temp_p;
}

/*The kernal function for Particles::applyForce.*/
static __global__ void particles_apply_force(float3 *v, float3 df, const int N){
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
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

static __global__ void particles_settle_kernal(float4 *p, float3 *v, float dt, const int N, const float Mx, const float My, const float Mz){
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
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
	if (px > Mx-1.001) {
		px = Mx-1.001;
//		v[tid].x = 0;
	}else if (px <1.001) {
		px = 1.001;
//		v[tid].x = 0;
	}

	py+=vy*dt;
	if (py > My-1.001) {
		py = My-1.001;
//		v[tid].y = 0;
	}else if (py < 1.001) {
		py = 1.001;
//		v[tid].y = 0;
	}

	pz+=vz*dt;
	if (pz > Mz-1.001) {
		pz = Mz-1.001;
//		v[tid].z = 0;
	}else if (pz < 1.001) {
		pz = 1.001;
//		v[tid].z = 0;
	}

	p[tid].x = px;
	p[tid].y = py;
	p[tid].z = pz;
}

void Particles::settle(float dt, float Mx, float My, float Mz){
	const int blockSize = 128;
	particles_settle_kernal<<<(_N-1)/blockSize+1, blockSize>>>(position.p(), velocity.p(), dt, _N, Mx, My, Mz);
}

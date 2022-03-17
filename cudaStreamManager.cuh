#pragma once
#ifndef CUDASTREAMMANAGER_CUH_BEFGBUXD
#define CUDASTREAMMANAGER_CUH_BEFGBUXD
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include "cudaErrorCheck.cuh"

namespace cudaR{
class cudaS
{
public:
	cudaS ();
	virtual ~cudaS ();
	void sync();
	bool query();
	cudaStream_t s();

private:
	cudaStream_t streamId;
};

cudaS::cudaS(){
	CHECK(cudaStreamCreate(&(this->streamId)));
}

cudaS::~cudaS(){
	CHECK(cudaStreamDestroy(this->streamId));
}

void cudaS::sync(){
	CHECK(cudaStreamSynchronize(this->streamId));
}

bool cudaS::query(){
	return cudaStreamQuery(this->streamId) == cudaSuccess;
}

cudaStream_t cudaS::s(){
	return this->streamId;
}

}
#endif /* end of include guard: CUDASTREAMMANAGER_CUH_BEFGBUXD */

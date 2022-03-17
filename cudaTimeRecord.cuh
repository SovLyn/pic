#pragma once
#ifndef CUDATIMERECORD_CUH_SWBMF4RB
#define CUDATIMERECORD_CUH_SWBMF4RB
#include <cuda.h>
#include <chrono>
#include "cudaErrorCheck.cuh"

namespace cudaR{

class cudaTimeRecord
{
public:
	cudaTimeRecord ();
	virtual ~cudaTimeRecord ();

	void start()const;
	void end()const;
	float t()const;

private:
	cudaEvent_t startEvent;
	cudaEvent_t endEvent;
};

cudaTimeRecord::cudaTimeRecord(){
	CHECK(cudaEventCreate(&startEvent));
	CHECK(cudaEventCreate(&endEvent));
}

cudaTimeRecord::~cudaTimeRecord(){
	CHECK(cudaEventDestroy(startEvent));
	CHECK(cudaEventDestroy(endEvent));
}

void cudaTimeRecord::start()const{
	CHECK(cudaEventRecord(startEvent));
	cudaEventQuery(startEvent);
}

void cudaTimeRecord::end()const{
	CHECK(cudaEventRecord(endEvent));
	CHECK(cudaEventSynchronize(endEvent));
}

float cudaTimeRecord::t()const{
	float toReturn;
	CHECK(cudaEventElapsedTime(&toReturn, startEvent, endEvent));
	return toReturn;
}

class chronoTimeRecord
{
public:
	chronoTimeRecord ();
	virtual ~chronoTimeRecord ();

	void start()const;
	void end()const;
	float t()const;

private:
	mutable std::chrono::time_point<std::chrono::steady_clock> startPoint, endPoint;
};

chronoTimeRecord::chronoTimeRecord(){}

chronoTimeRecord::~chronoTimeRecord(){}

__inline__ void chronoTimeRecord::start()const{
	startPoint = std::chrono::steady_clock::now();
}

__inline__ void chronoTimeRecord::end()const{
	endPoint = std::chrono::steady_clock::now();
}

float chronoTimeRecord::t()const{
	return std::chrono::duration_cast<std::chrono::nanoseconds>(endPoint-startPoint).count()/static_cast<float>(1e6);
}

}

#endif /* end of include guard: CUDATIMERECORD_CUH_SWBMF4RB */

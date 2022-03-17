#pragma once
#ifndef CUDARESOURCEMANAGER_CUH_CCVOUDKT
#define CUDARESOURCEMANAGER_CUH_CCVOUDKT

#include <cuda.h>
#include <cuda_runtime.h>
#include "cudaErrorCheck.cuh"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <boost/core/noncopyable.hpp>
#include <boost/shared_array.hpp>
namespace cudaR{

typedef unsigned int u;

template<typename T>
class cudaM:private boost::noncopyable{
	/*
	   YOU SHALL NEVER FREE cudaM.p() BY YOURSELF.
	 */
public:
	explicit cudaM (u l=1);
	virtual ~cudaM ();

	T* p()const;
	u L()const;
	void copyTo(T* to, u copyNum=0, u offset=0)const;
	void copyFrom(const T* from, u copyNum=0, u offset=0)const;
	void copyTo(boost::shared_array<T>& to, u copyNum=0, u offset=0)const;
	void copyFrom(const boost::shared_array<T>& from, u copyNum=0, u offset=0)const;
	void setVal(int val=0);

private:
	const u LENGTH;
	T* cudaP;
};

template<typename T>
cudaM<T>::cudaM(u l):
LENGTH(l){
	CHECK(cudaMalloc(&cudaP, LENGTH*sizeof(T)));
	CHECK(cudaMemset(cudaP, 0, LENGTH*sizeof(T)));
}

template<typename T>
cudaM<T>::~cudaM(){
	CHECK(cudaFree(cudaP));
}

template<typename T>
T* cudaM<T>::p()const{
	return cudaP;
}

template<typename T>
u cudaM<T>::L()const{
	return LENGTH;
}

template<typename T>
void cudaM<T>::copyTo(T* to, u copyNum, u offset)const{
	if(copyNum == 0)copyNum = LENGTH;
	CHECK(cudaMemcpy(to, cudaP+offset, copyNum*sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
void cudaM<T>::copyFrom(const T* from, u copyNum, u offset)const{
	if(copyNum==0)copyNum = LENGTH;
	CHECK(cudaMemcpy(cudaP+offset, from, copyNum*sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void cudaM<T>::copyTo(boost::shared_array<T>& to, u copyNum, u offset)const{
	if(copyNum == 0)copyNum = LENGTH;
	CHECK(cudaMemcpy(to.get(), cudaP+offset, copyNum*sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
void cudaM<T>::copyFrom(const boost::shared_array<T>& from, u copyNum, u offset)const{
	if(copyNum==0)copyNum = LENGTH;
	CHECK(cudaMemcpy(cudaP+offset, from.get(), copyNum*sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void cudaM<T>::setVal(int val){
	CHECK(cudaMemset(cudaP, val, sizeof(T)*LENGTH));
}

template<typename T>
class cudaUM:private boost::noncopyable{
	/*
	   YOU SHALL NEVER FREE cudaUM.p() BY YOURSELF.
	 */
public:
	explicit cudaUM (u l=1, unsigned flags = cudaMemAttachGlobal);
	virtual ~cudaUM ();

	T* p()const;
	u L()const;
	void copyTo(T* to, u copyNum=0, u offset=0)const;
	void copyFrom(const T* from, u copyNum=0, u offset=0)const;
	void copyTo(boost::shared_array<T>& to, u copyNum=0, u offset=0)const;
	void copyFrom(const boost::shared_array<T>& from, u copyNum=0, u offset=0)const;
	void setVal(int val=0);
	void prefetch(int count=-1, int dstDevice = 0, cudaStream_t stream = NULL);

private:
	const u LENGTH;
	T* cudaP;
};

template<typename T>
cudaUM<T>::cudaUM(u l, unsigned flags):
LENGTH(l){
	CHECK(cudaMallocManaged(&cudaP, LENGTH*sizeof(T), flags));
	CHECK(cudaMemset(cudaP, 0, LENGTH*sizeof(T)));
}

template<typename T>
cudaUM<T>::~cudaUM(){
	CHECK(cudaFree(cudaP));
}

template<typename T>
T* cudaUM<T>::p()const{
	return cudaP;
}

template<typename T>
u cudaUM<T>::L()const{
	return LENGTH;
}

template<typename T>
void cudaUM<T>::copyTo(T* to, u copyNum, u offset)const{
	if(copyNum == 0)copyNum = LENGTH;
	CHECK(cudaMemcpy(to, cudaP+offset, copyNum*sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
void cudaUM<T>::copyFrom(const T* from, u copyNum, u offset)const{
	if(copyNum==0)copyNum = LENGTH;
	CHECK(cudaMemcpy(cudaP+offset, from, copyNum*sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void cudaUM<T>::copyTo(boost::shared_array<T>& to, u copyNum, u offset)const{
	if(copyNum == 0)copyNum = LENGTH;
	CHECK(cudaMemcpy(to.get(), cudaP+offset, copyNum*sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
void cudaUM<T>::copyFrom(const boost::shared_array<T>& from, u copyNum, u offset)const{
	if(copyNum==0)copyNum = LENGTH;
	CHECK(cudaMemcpy(cudaP+offset, from.get(), copyNum*sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void cudaUM<T>::setVal(int val){
	CHECK(cudaMemset(cudaP, val, sizeof(T)*LENGTH));
}

template<typename T>
void cudaUM<T>::prefetch(int count, int dstDevice, cudaStream_t stream){
	if(count<1)count=LENGTH;
	CHECK(cudaMemPrefetchAsync(cudaP, count*sizeof(T), dstDevice, stream));
}
}

#endif /* end of include guard: CUDARESOURCEMANAGER_CUH_CCVOUDKT */

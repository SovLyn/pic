#pragma once
#ifndef CUDAERRORCHECK_CUH_OLE09MVW
#define CUDAERRORCHECK_CUH_OLE09MVW

#include <iostream>
#ifndef CHECK
#define CHECK(_call) {\
	const cudaError_t _code = _call;\
	if(_code!=cudaSuccess){\
		std::cout<<"Error happened in: "\
		<<#_call\
		<<"at line "<<__LINE__\
		<<" ----> "\
		<<cudaGetErrorString(_code)\
		<<std::endl;\
	}\
}
#endif


#endif /* end of include guard: CUDAERRORCHECK_CUH_OLE09MVW */

/*
   nvcc -lGL -lGLU -lglut -lglfw %
 */
#include <GL/freeglut_std.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glu.h>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <chrono>

#include <GLFW/glfw3.h>
#include <GLES3/gl3.h>
#include <GL/freeglut.h>
#include <type_traits>
#include <cuda_gl_interop.h>

#define P(x) std::cout<<#x<<" = "<<(x)<<std::endl

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


typedef unsigned int u;

const int windowWidth = 800;
const int windowHeight = 800;
const unsigned int dataNx = 320;
const unsigned int dataNy = 320;
const bool useGPU = true;
float4 *data = 0;

auto startTime = std::chrono::high_resolution_clock::now();
const int refreshDelay = 1;

int mouseButton=0;
int mouseTx=0;
int mouseTy=0;
float rotateX=0.0;
float rotateY=0.0;
float translateZ=-3.0;
int frames = 0;

unsigned int vbo;
struct cudaGraphicsResource *cudaVboRes;

void calcDataCPU(float4 data[], unsigned int Nx, unsigned int Ny, float t);
void calcDataGPU(float4 data[], unsigned int Nx, unsigned int Ny, float t);
void keyboardOP(unsigned char key, int, int);
void display();
void timerEvent(int);
void cleanup();
void mouseOP(int button, int state, int x, int y);
void motionOP(int x, int y);
void createVBO(unsigned int *vbo, struct cudaGraphicsResource **cvr, unsigned int flag);
void deleteVBO(unsigned int *vbo, struct cudaGraphicsResource **cvr);
void clean();
void runCuda(float t);

int main(int argc, char *argv[])
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(windowWidth, windowHeight);
	glutCreateWindow("cuda & OpenGL");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboardOP);
	glutMouseFunc(mouseOP);
	glutMotionFunc(motionOP);
	glutTimerFunc(refreshDelay, timerEvent, 0);
	glutCloseFunc(clean);

	glClearColor(0.4, 0.0, 0.7, 1.0);
	glDisable(GL_DEPTH_TEST);
	glViewport(0, 0, windowWidth, windowHeight);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, static_cast<float>(windowWidth)/static_cast<float>(windowHeight), 0.1, 10.0);

	createVBO(&vbo, &cudaVboRes, cudaGraphicsMapFlagsWriteDiscard);
	if(!useGPU){
		data = new float4[dataNx*dataNy];
	}

	runCuda(0);
	startTime = std::chrono::high_resolution_clock::now();
	glutMainLoop();
	
}

void calcDataCPU(float4 data[], unsigned int Nx, unsigned int Ny, float t){
	for (int i = 0; i < Nx; ++i) {
		for (int j = 0; j < Ny; ++j) {
			float x = (i/float(Nx-1)*2 - 1);
			float y = (j/float(Ny-1)*2 - 1);
			data[i*Nx + j] = make_float4(x, y, cosf(t*M_PI/2)*cosf(x*M_PI)*cosf(y*M_PI), 1.0f);
		}
	}
}

__global__ void simpleGPUkernel(float4 *data, unsigned int Nx, unsigned int Ny, float t){
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y;
	float x = (idx/float(Nx-1)*2 - 1);
	float y = (idy/float(Ny-1)*2 - 1);
	float z = cosf((x+t/4)*M_PI)*cosf((1.5*y+t/4)*M_PI)+cosf((0.5*x+t/4)*M_PI)*cosf((y+t/4)*M_PI);;
	z/=2;
	data[idx*Nx+idy] = make_float4(x, z, y, 1.0f);
//	printf("idx:%d, idy:%d, x:%f, y:%f, z:%f\n", idx, idy, x, y, z);
}

void calcDataGPU(float4 *data, unsigned int Nx, unsigned int Ny, float t){
	dim3 blockD(8, 8, 1);
	dim3 gridD((Nx-1)/blockD.x+1, (Ny-1)/blockD.y+1, 1);
	simpleGPUkernel<<<gridD, blockD>>>(data, Nx, Ny, t);
}

void keyboardOP(unsigned char key, int a, int b){
	switch (key) {
		case 27:
			glutDestroyWindow(glutGetWindow());
			return;
		default:
		return;
	}
}

void timerEvent(int value){
	if (glutGetWindow()){
		glutPostRedisplay();
		glutTimerFunc(refreshDelay, timerEvent,0);
		std::chrono::duration<float> timePast = std::chrono::high_resolution_clock::now()-startTime;
		std::cout << "fps:"<<static_cast<float>(frames)/(timePast.count())<<'\r';
	}
	return;
}

void mouseOP(int button, int state, int x, int y){
	if(state==GLUT_DOWN){
		mouseButton|=1<<button;
	}else if(state == GLUT_UP){
		mouseButton=0;
	}
	mouseTx = x;
	mouseTy = y;
	return;
}

void motionOP(int x, int y){
	float dx = static_cast<float>(x-mouseTx);
	float dy = static_cast<float>(y-mouseTy);

	if(mouseButton&1){
		rotateX += dx*0.2f;
		rotateY += dy*0.2f;
	}else if(mouseButton&4){
		translateZ+=dy*0.01;
	}
	mouseTx = x;
	mouseTy = y;
	return;
}

void runCuda(float t){
	float4 *dptr;
	CHECK(cudaGraphicsMapResources(1, &cudaVboRes, 0));
	size_t numBytes;
	CHECK(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &numBytes, cudaVboRes));
	calcDataGPU(dptr, dataNx, dataNy, t);
	CHECK(cudaGraphicsUnmapResources(1, &cudaVboRes, 0));
}

void display(){
	std::chrono::duration<float> timePast = std::chrono::high_resolution_clock::now()-startTime;
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	if(!useGPU){
		calcDataCPU(data, dataNx, dataNy, timePast.count());
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_DYNAMIC_DRAW);
	}else{
		runCuda(timePast.count());
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//	P(translateZ);
//	P(rotateX);
//	P(rotateY);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translateZ);
	glRotatef(rotateY, 1.0, 0.0, 0.0);
	glRotatef(rotateX, 0.0, 1.0, 0.0);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(0.9, 1.0, 0.0);
	glPointSize(2);
	glDrawArrays(GL_POINTS, 0, dataNx*dataNy);
	frames++;
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();

}

void createVBO(unsigned int *vbo, struct cudaGraphicsResource **cvr, unsigned int flag){
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	if(useGPU){
		glBufferData(GL_ARRAY_BUFFER, dataNx*dataNy*sizeof(float4), 0, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		CHECK(cudaGraphicsGLRegisterBuffer(cvr, *vbo, flag));
	}
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return;
}

void deleteVBO(unsigned int *vbo, struct cudaGraphicsResource *cvr){
	if(useGPU){
		CHECK(cudaGraphicsUnregisterResource(cvr));
	}
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
	return;
}

void clean(){
	if(vbo){
		deleteVBO(&vbo, cudaVboRes);
	}
}

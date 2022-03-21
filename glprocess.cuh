#pragma once
#ifndef GLPROCESS_CUH

#define GLPROCESS_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#include <GL/freeglut_std.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glu.h>
#include <chrono>

#include <GLFW/glfw3.h>
#include <GLES3/gl3.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include "MAC.cuh"

extern const unsigned int particleNum;

Particles& getParticles();

extern const int windowWidth;
extern const int windowHeight;
extern const int refreshDelay;

extern decltype(std::chrono::high_resolution_clock::now()) startTime;

extern Particles& p;

extern int mouseButton;
extern int mouseTx;
extern int mouseTy;
extern float rotateX;
extern float rotateY;
extern float translateZ;
extern int frames;
extern unsigned int vbo;
extern struct cudaGraphicsResource *cudaVboRes;

void keyboardOP(unsigned char key, int, int);
void display();
void timerEvent(int);
void cleanup();
void mouseOP(int button, int state, int x, int y);
void motionOP(int x, int y);
void createVBO(unsigned int *vbo, struct cudaGraphicsResource **cvr, unsigned int flag);
void deleteVBO(unsigned int *vbo, struct cudaGraphicsResource **cvr);
void clean();
void fluidUpdate(float t);

#endif /* end of include guard: GLPROCESS_CUH */

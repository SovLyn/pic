#include "glprocess.cuh"

extern const unsigned int particleNum = 5*7*3;
const int windowWidth = 800;
const int windowHeight = 800;
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

decltype(std::chrono::high_resolution_clock::now()) startTime = std::chrono::high_resolution_clock::now();

Particles& getParticles(){
	static Particles p(particleNum);
	return p;
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


void display(){
	std::chrono::duration<float> timePast = std::chrono::high_resolution_clock::now()-startTime;
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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
	glDrawArrays(GL_POINTS, 0, particleNum);
	frames++;
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();

	fluidUpdate(0.0);
}

void createVBO(unsigned int *vbo, struct cudaGraphicsResource **cvr, unsigned int flag){
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	glBufferData(GL_ARRAY_BUFFER, particleNum*sizeof(float4), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	CHECK(cudaGraphicsGLRegisterBuffer(cvr, *vbo, flag));
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return;
}

void deleteVBO(unsigned int *vbo, struct cudaGraphicsResource *cvr){
	CHECK(cudaGraphicsUnregisterResource(cvr));
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

void fluidUpdate(float t){
	float4 *dptr;
	size_t numBytes;
	CHECK(cudaGraphicsMapResources(1, &cudaVboRes, 0))
	CHECK(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &numBytes, cudaVboRes));

	CHECK(cudaMemcpy(dptr, getParticles().position.p(), getParticles().N()*sizeof(float4), cudaMemcpyDeviceToDevice));
	CHECK(cudaGraphicsUnmapResources(1, &cudaVboRes, 0));
}

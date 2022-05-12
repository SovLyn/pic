#include "glprocess.cuh"
#include <GL/gl.h>
#include <GLES3/gl3.h>
#include <glm/ext/matrix_transform.hpp>

const int windowWidth = 1000;
const int windowHeight = 1000;
const int refreshDelay = 1;

int mouseButton=0;
int mouseTx=0;
int mouseTy=0;
float rotateX=0.0;
float rotateY=0.0;
float translateZ=1.1;
int frames = 0;

std::function<void(float, float, MAC&)> forceFunc=nullptr;

glm::mat4 view = glm::lookAt(glm::vec3(1.0f, 1.0f, 1.0f),
							 glm::vec3(0.5f, 0.5f, 0.5f),
							 glm::vec3(0.0f, 0.0f, 1.0f));
glm::mat4 projection = glm::perspective(glm::radians(75.f), 1.f, 0.1f, 1000.f); 
unsigned int view_loc;
unsigned int projection_loc;

unsigned int VBO[2]={0,};
unsigned int VAO[2]={0,};
struct cudaGraphicsResource *cudaVboRes;

decltype(std::chrono::high_resolution_clock::now()) startTime = std::chrono::high_resolution_clock::now();

static unsigned int compile_shader(unsigned int type, const std::string& source){
	unsigned int id = glCreateShader(type);
	const char* src = source.c_str();
	glShaderSource(id, 1, &src, nullptr);
	glCompileShader(id);

	int result;
	glGetShaderiv(id, GL_COMPILE_STATUS, &result);

	//Error checking
	if(result == GL_FALSE){
		 int length;
		 glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
		 P(length);
		 char *message = new char[length];
		 glGetShaderInfoLog(id, length, &length, message);
		 std::cout << "Fail to compile "<<(type == GL_VERTEX_SHADER?"vertex shader":"fragment shader")<<"!\nINFO:" << std::endl;
		 std::cout << message << std::endl;
		 std::cout << (type == GL_VERTEX_SHADER?"vertex shader":"fragment shader")<<":\n"
			<<source << std::endl;

		 glDeleteShader(id);
		 delete[] message;
		 return 0;
	}

	return id;
}

unsigned int create_shader(const std::string& vertex_shader, const std::string& fragment_shader){
	unsigned int program = glCreateProgram();
	unsigned int vs = compile_shader(GL_VERTEX_SHADER, vertex_shader);
	unsigned int fs = compile_shader(GL_FRAGMENT_SHADER, fragment_shader);

	glAttachShader(program, vs);	
	glAttachShader(program, fs);
	glLinkProgram(program);
	glValidateProgram(program);

	glDeleteShader(vs);
	glDeleteShader(fs);

	return program;
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
		rotateX += dx*0.02f;
		rotateY += dy*0.02f;
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

	glm::vec3 center = glm::vec3(float(getMAC().Nx()), float(getMAC().Ny()), float(getMAC().Nz()));
	glm::vec3 eye = center*translateZ;
	glm::mat4 rotateMat(1);
	rotateMat = glm::rotate(rotateMat, -rotateX, glm::vec3(0.0f, 0.0f, 1.0f));
	rotateMat = glm::rotate(rotateMat, rotateY, glm::vec3(0.0f, 1.0f, 0.0f));
	eye = glm::vec3(rotateMat*glm::vec4(eye, 1.0f));
	view = glm::lookAt(eye+center, 
					   center,
					   glm::vec3(0.0f, 0.0f, 1.0f));
	glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm::value_ptr(view));
	glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm::value_ptr(projection));

	glBindVertexArray(VAO[0]);
	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glEnableVertexAttribArray(0);
//	glVertexPointer(4, GL_FLOAT, 0, 0);

//	glEnableClientState(GL_VERTEX_ARRAY);
//	glColor3f(0.9, 1.0, 0.0);
	glPointSize(1);
	glDrawArrays(GL_POINTS, 0, particleNum);
	frames++;
//	glDisableClientState(GL_VERTEX_ARRAY);

	glBindVertexArray(VAO[1]);
	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
	glEnableVertexAttribArray(1);
	glDrawArrays(GL_TRIANGLES, 0, 12);

	glutSwapBuffers();

	fluidUpdate(timePast.count(), 1.0/60.0, forceFunc);
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
	if(VBO[0]){
		deleteVBO(VBO, cudaVboRes);
	}
}

void fluidUpdate(float t, float delta_t, std::function<void(float, float, MAC&)> forceFunc){
//	printf("\nframes: %d\n", frames);
	float4 *dptr;
	size_t numBytes;

	CHECK(cudaGraphicsMapResources(1, &cudaVboRes, 0))
	CHECK(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &numBytes, cudaVboRes));
	int Nx = getMAC().Nx();
	int Ny = getMAC().Ny();
	int Nz = getMAC().Nz();

	getMAC().particlesToGrid(getParticles());
//	getParticles().applyForce(make_float3(0.00, 0.00, -0.01), t);
	if (forceFunc==nullptr) {
		getMAC().applyForce(make_float3(0.00, 0.00, -15), delta_t);
	}else{
		forceFunc(t, delta_t, getMAC());
	}

	getMAC().solvePressure(80, 1.0);
	getMAC().gridToParticles(getParticles());

	getParticles().settle(delta_t, float(getMAC().Nx()), float(getMAC().Ny()), float(getMAC().Nz()));


	CHECK(cudaMemcpy(dptr, getParticles().position.p(), getParticles().N()*sizeof(float4), cudaMemcpyDeviceToDevice));
	CHECK(cudaGraphicsUnmapResources(1, &cudaVboRes, 0));

//	if (frames > 200) {
//		exit(1);
//	}
}


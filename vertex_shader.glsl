#version 410 core

layout (location = 0) in vec4 a_position;

uniform mat4 view;
uniform mat4 projection;

void main(){
	gl_Position = projection*view*a_position;
//	gl_Position = view*a_position;
}

#version 410 core
#extension GL_ARB_explicit_uniform_location : require
layout(location = 0) in vec4 coord4d;
layout(location = 1) uniform mat4 mvp_matrix;

void main(){
	gl_PointSize = 1.0;
	gl_Position = mvp_matrix * coord4d;
}